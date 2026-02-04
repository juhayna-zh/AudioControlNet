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
    """Remove [s,e) from the list of available intervals and return the updated list of available intervals."""
    new = []
    for a,b in avail:
        if e <= a or s >= b:
            # no overlap
            new.append((a,b))
        else:
            # with overlap
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
    Generate a random multi-segment mask (an indefinite-length list, with each segment being (start, end), in seconds).

    Args:
        duration (float): Total audio duration in seconds, must be > 0.
        min_seg (float): Minimum length of each segment in seconds.
        max_seg (float): Maximum length of each segment in seconds.
        min_coverage (float): Minimum coverage ratio (0..1).
        max_coverage (float): Maximum coverage ratio (0..1).
        min_segments (int): Minimum number of segments (>=1).
        max_segments (int): Maximum number of segments (>= min_segments).
        allow_overlap (bool, optional): Whether to allow overlap between segments (default False).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        List of (start, end) tuples sorted by time. Returns an empty list if generation fails
        (e.g., invalid duration).
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

    # Randomly select the number of segments & total coverage rate (in seconds)
    num_segments = rng.randint(min_segments, max_segments)
    target_frac = rng.uniform(min_coverage, max_coverage)
    target_mask_total = target_frac * duration

    # If target_mask_total is very small, directly return null
    if target_mask_total <= 1e-12:
        return []

    # Ensure that the total coverage can be achieved under the constraints of segment count and length
    # If it is not possible (e.g., num_segments * min_seg > target), try reducing the segment count until it becomes feasible
    while num_segments > 1 and num_segments * min_seg > target_mask_total:
        num_segments -= 1
    # If it still doesn't work, increase the target to the minimum feasible value (avoiding negative values)
    if num_segments * min_seg > target_mask_total:
        target_mask_total = num_segments * min_seg

    # Similarly, ensure that target does not exceed num_segments * max_seg
    if target_mask_total > num_segments * max_seg:
        # If it exceeds, reduce num_segments and try to adapt (until num_segments == 1)
        while num_segments > 1 and target_mask_total > num_segments * max_seg:
            num_segments -= 1
        # If the time limit is still exceeded, the time will be clipped from target_clip to num_segments * max_seg
        target_mask_total = min(target_mask_total, num_segments * max_seg)

    # Allocate the length of each segment (using a step-by-step allocation method to ensure that each falls within [min_seg, max_seg] and the total is appr target_mask_total)
    lengths = []
    allocated = 0.0
    for i in range(num_segments):
        rem = num_segments - i
        # Calculate the min/max values allowed for the current segment
        min_possible = max(min_seg, target_mask_total - allocated - (rem - 1) * max_seg)
        max_possible = min(max_seg, target_mask_total - allocated - (rem - 1) * min_seg)
        if max_possible < min_possible:
            # Align the two values when they are inconsistent to avoid error
            max_possible = min_possible
        seg_len = rng.uniform(min_possible, max_possible)
        lengths.append(seg_len)
        allocated += seg_len

    # Place each segment (when not overlapping, select from the available intervals)
    segments: List[Tuple[float,float]] = []
    if allow_overlap:
        for L in lengths:
            start = rng.uniform(0.0, max(0.0, duration - L))
            segments.append((start, start + L))
    else:
        # First, arrange the segments to be placed in descending order of length
        lengths.sort(reverse=True)
        avail = [(0.0, duration)]  # avaliable interval list
        for L in lengths:
            placed = False
            # Try randomly selecting intervals
            attempts = 0
            max_attempts = 50
            while attempts < max_attempts and not placed:
                attempts += 1
                # Calculate the length of each available interval, and randomly select an interval based on its length-weighted probability
                total_avail = sum(b - a for a,b in avail)
                if total_avail < L - 1e-9:
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
                # Fallback: Try to fit this segment into the largest available interval (reduce its length to fit)
                if not avail:
                    break
                # Find the longest available interval
                a,b = max(avail, key=lambda x: x[1]-x[0])
                max_fit = b - a
                if max_fit < min_seg - 1e-9:
                    # The maximum gap is too small to accommodate the smallest segment, jump out
                    break
                L2 = min(L, max_fit)
                start = rng.uniform(a, b - L2)
                end = start + L2
                segments.append((start, end))
                avail = _remove_interval(avail, start, end)

    # Final: clip into the boundary, sort by time, and round (with 3 decimal)
    final = []
    for s,e in segments:
        s = max(0.0, min(duration, s))
        e = max(0.0, min(duration, e))
        if e - s > 1e-6:
            final.append((round(s, 3), round(e, 3)))

    final.sort(key=lambda x: x[0])
    return final
