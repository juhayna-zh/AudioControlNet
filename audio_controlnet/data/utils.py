import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Optional, Union, List, Tuple

import torch
import torch.distributed as dist
from tensordict import MemoryMappedTensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from audio_controlnet.utils.dist_utils import local_rank, world_size

scratch_path = Path(os.environ['SLURM_SCRATCH'] if 'SLURM_SCRATCH' in os.environ else '/dev/shm')
shm_path = Path('/dev/shm')

log = logging.getLogger()


def reseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def local_scatter_torch(obj: Optional[Any]):
    if world_size == 1:
        # Just one worker. Do nothing.
        return obj

    array = [obj] * world_size
    target_array = [None]
    if local_rank == 0:
        dist.scatter_object_list(target_array, scatter_object_input_list=array, src=0)
    else:
        dist.scatter_object_list(target_array, scatter_object_input_list=None, src=0)
    return target_array[0]


class ShardDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.shards = sorted(os.listdir(root))

    def __len__(self):
        return len(self.shards)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.root, self.shards[idx]), weights_only=True)


def get_tmp_dir(in_memory: bool) -> Path:
    return shm_path if in_memory else scratch_path


def load_shards_and_share(data_path: Union[str, Path], ids: list[int],
                          in_memory: bool) -> MemoryMappedTensor:
    if local_rank == 0:
        with tempfile.NamedTemporaryFile(prefix='shared-tensor-', dir=get_tmp_dir(in_memory)) as f:
            log.info(f'Loading shards from {data_path} into {f.name}...')
            data = load_shards(data_path, ids=ids, tmp_file_path=f.name)
            data = share_tensor_to_all(data)
            torch.distributed.barrier()
            f.close()  # why does the context manager not close the file for me?
    else:
        log.info('Waiting for the data to be shared with me...')
        data = share_tensor_to_all(None)
        torch.distributed.barrier()

    return data


def load_shards(
    data_path: Union[str, Path],
    ids: list[int],
    *,
    tmp_file_path: str,
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:

    id_set = set(ids)
    shards = sorted(os.listdir(data_path))
    log.info(f'Found {len(shards)} shards in {data_path}.')
    first_shard = torch.load(os.path.join(data_path, shards[0]), weights_only=True)

    log.info(f'Rank {local_rank} created file {tmp_file_path}')
    first_item = next(iter(first_shard.values()))
    log.info(f'First item shape: {first_item.shape}')
    mm_tensor = MemoryMappedTensor.empty(shape=(len(ids), *first_item.shape),
                                         dtype=torch.float32,
                                         filename=tmp_file_path,
                                         existsok=True)
    total_count = 0
    used_index = set()
    id_indexing = {i: idx for idx, i in enumerate(ids)}
    # faster with no workers; otherwise we need to set_sharing_strategy('file_system')
    loader = DataLoader(ShardDataset(data_path), batch_size=1, num_workers=0)
    for data in tqdm(loader, desc='Loading shards'):
        for i, v in data.items():
            if i not in id_set:
                continue

            # tensor_index = ids.index(i)
            tensor_index = id_indexing[i]
            if tensor_index in used_index:
                raise ValueError(f'Duplicate id {i} found in {data_path}.')
            used_index.add(tensor_index)
            mm_tensor[tensor_index] = v
            total_count += 1

    assert total_count == len(ids), f'Expected {len(ids)} tensors, got {total_count}.'
    log.info(f'Loaded {total_count} tensors from {data_path}.')

    return mm_tensor


def share_tensor_to_all(x: Optional[MemoryMappedTensor]) -> MemoryMappedTensor:
    """
    x: the tensor to be shared; None if local_rank != 0
    return: the shared tensor
    """

    # there is no need to share your stuff with anyone if you are alone; must be in memory
    if world_size == 1:
        return x

    if local_rank == 0:
        assert x is not None, 'x must not be None if local_rank == 0'
    else:
        assert x is None, 'x must be None if local_rank != 0'

    if local_rank == 0:
        filename = x.filename
        meta_information = (filename, x.shape, x.dtype)
    else:
        meta_information = None

    filename, data_shape, data_type = local_scatter_torch(meta_information)
    if local_rank == 0:
        data = x
    else:
        data = MemoryMappedTensor.from_filename(filename=filename,
                                                dtype=data_type,
                                                shape=data_shape)

    return data


def _remove_interval(avail: List[Tuple[float,float]], s: float, e: float):
    """从可用区间列表中移除 [s,e)，返回新的可用区间列表。"""
    new = []
    for a,b in avail:
        if e <= a or s >= b:
            # 无交集
            new.append((a,b))
        else:
            # 有交集，可能留下左右两段
            if s > a:
                new.append((a, s))
            if e < b:
                new.append((e, b))
    return new

def random_mask_segments(
    duration: float,
    min_seg: float = 0.2,
    max_seg: float = 1.0,
    min_coverage: float = 0.0,
    max_coverage: float = 0.3,
    min_segments: int = 1,
    max_segments: int = 5,
    allow_overlap: bool = False,
    seed: Optional[int] = None,
) -> List[Tuple[float,float]]:
    """
    生成随机多段 mask（不定长列表，每段为 (start, end)，单位：秒）。

    Args:
        duration: 音频总时长（秒），必须 > 0。
        min_seg: 每段最短长度（秒）。
        max_seg: 每段最长长度（秒）。
        min_coverage: 最小遮盖比例（0..1）。
        max_coverage: 最大遮盖比例（0..1）。
        min_segments: 最少段数（>=1）。
        max_segments: 最多段数（>= min_segments）。
        allow_overlap: 是否允许段之间重叠（默认 False）。
        seed: 随机种子（可选，用于复现）。

    Returns:
        按时间排序的 (start, end) 列表。若无法生成（例如 duration 非法），返回空列表。
    """
    if duration <= 0:
        return []

    rng = random.Random(seed)

    # clamp and sanitize inputs
    min_seg = max(0.0, min_seg)
    max_seg = max(min_seg, min(max_seg, duration))
    min_coverage = max(0.0, min(1.0, min_coverage))
    max_coverage = max(min_coverage, min(1.0, max_coverage))
    min_segments = max(1, int(min_segments))
    max_segments = max(min_segments, int(max_segments))

    # 随机选择段数 & 总遮盖率（以秒为单位）
    num_segments = rng.randint(min_segments, max_segments)
    target_frac = rng.uniform(min_coverage, max_coverage)
    target_mask_total = target_frac * duration

    # 若 target_mask_total 很小且为 0，直接返回空
    if target_mask_total <= 1e-12:
        return []

    # 保证在段数和长度约束下可实现总遮盖量
    # 如果不可能（例如 num_segments * min_seg > target），尝试缩小段数直到可行
    while num_segments > 1 and num_segments * min_seg > target_mask_total:
        num_segments -= 1
    # 若仍然不行，把 target 提到最小可行值（避免负值）
    if num_segments * min_seg > target_mask_total:
        target_mask_total = num_segments * min_seg

    # 同理，保证 target 不超过 num_segments * max_seg
    if target_mask_total > num_segments * max_seg:
        # 如果超出，降低 num_segments 尝试适配（直到 num_segments==1）
        while num_segments > 1 and target_mask_total > num_segments * max_seg:
            num_segments -= 1
        # 最后仍超出时，将 target_clip 到 num_segments * max_seg
        target_mask_total = min(target_mask_total, num_segments * max_seg)

    # 分配每段长度（逐步分配法，确保每段在 [min_seg, max_seg] 且总和约等于 target_mask_total）
    lengths = []
    allocated = 0.0
    for i in range(num_segments):
        rem = num_segments - i
        # 为当前段计算允许的最小/最大值，保证剩余段能满足约束
        min_possible = max(min_seg, target_mask_total - allocated - (rem - 1) * max_seg)
        max_possible = min(max_seg, target_mask_total - allocated - (rem - 1) * min_seg)
        if max_possible < min_possible:
            # 当数值不一致时把两者对齐，避免异常
            max_possible = min_possible
        seg_len = rng.uniform(min_possible, max_possible)
        lengths.append(seg_len)
        allocated += seg_len

    # 放置每段（不重叠时要从可用区间挑选）
    segments: List[Tuple[float,float]] = []
    if allow_overlap:
        for L in lengths:
            start = rng.uniform(0.0, max(0.0, duration - L))
            segments.append((start, start + L))
    else:
        # 先把要放的段按长度降序排列（先放长段更容易成功）
        lengths.sort(reverse=True)
        avail = [(0.0, duration)]  # 可用区间列表
        for L in lengths:
            placed = False
            # 尝试若干次随机选区间放置
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts and not placed:
                attempts += 1
                # 计算每个可用区间的长度，按长度加权随机选区间
                total_avail = sum(b - a for a,b in avail)
                if total_avail < L - 1e-9:
                    # 空间不足直接跳出尝试
                    break
                r = rng.random() * total_avail
                acc = 0.0
                chosen = None
                for (a,b) in avail:
                    seglen = b - a
                    if acc + seglen >= r:
                        chosen = (a,b)
                        break
                    acc += seglen
                if chosen is None:
                    chosen = avail[-1]
                a,b = chosen
                if b - a >= L - 1e-12:
                    start = rng.uniform(a, b - L)
                    end = start + L
                    segments.append((start, end))
                    avail = _remove_interval(avail, start, end)
                    placed = True
                else:
                    # this interval too small, try another loop
                    continue

            if not placed:
                # 退路：尝试把这个段塞入最大可用区间（缩短长度以适配）
                if not avail:
                    # 没有可用区间了，跳过剩余段
                    break
                # 找到最长可用区间
                a,b = max(avail, key=lambda x: x[1]-x[0])
                max_fit = b - a
                if max_fit < min_seg - 1e-9:
                    # 最大间隙都太小，放不下最小段，跳出
                    break
                L2 = min(L, max_fit)
                start = rng.uniform(a, b - L2)
                end = start + L2
                segments.append((start, end))
                avail = _remove_interval(avail, start, end)

    # 最后整理：剪入边界，按时间排序并四舍五入（保留 3 位小数）
    final = []
    for s,e in segments:
        s = max(0.0, min(duration, s))
        e = max(0.0, min(duration, e))
        if e - s > 1e-6:
            final.append((round(s, 3), round(e, 3)))

    final.sort(key=lambda x: x[0])
    return final
