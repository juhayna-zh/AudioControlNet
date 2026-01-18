import logging
from pathlib import Path
from typing import Union, Optional, List, Any
import json
import random

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from audio_controlnet.utils.dist_utils import local_rank
import numpy as np
import glob
import torch.nn.functional as F

from .extract_cond import extract_all_conditions

log = logging.getLogger()

def custom_collate_fn(batch):
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    collated = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if key in ('events', 'inpaint', 'remove', 'reference', 'insert', 'insert_events'):
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


class ExtractedAudio(Dataset):  
    def __init__(
        self,
        metadata_path: Union[str, Path],
        *,
        concat_text_fc: bool, 
        npz_dir: Union[str, Path],
        data_dim: dict[str, int],
        repa_npz_dir: Optional[Union[str, Path]],   # if passed, repa features (zs) would be returned
        exclude_cls: Optional[bool], 
        repa_version: Optional[int], 
        control_types: Optional[List[str]] = None,
        conditioners_config: Optional[Any] = None,
        background_metadata: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.df_list = self.read_metadata(metadata_path)
        try:
            self.ids = [str(d['id']) for d in self.df_list]
        except:
            self.ids = [str(d['key']) for d in self.df_list]
        npz_files = glob.glob(f"{npz_dir}/*.npz")
        self.concat_text_fc = concat_text_fc
        self.exclude_cls = exclude_cls
        self.repa_version = repa_version
        self.control_types = control_types
        self.conditioners_config = conditioners_config
        self.background_metadata = self.read_metadata(background_metadata)
        
        self.empty_string_feat = torch.load('./weights/empty_string_t5.pth', weights_only=True)[0]  # abandon the first (btz) dim. 
        self.empty_string_feat_c = torch.load('./weights/empty_string_clap_c.pth',  weights_only=True)[0]
    
        if self.concat_text_fc: 
            log.info(f'We will concat the pooled text_features and text_features_c for text condition')

        # dimension check
        sample = np.load(f'{npz_dir}/0.npz')  
        mean_s = [len(npz_files)] + list(sample['mean'].shape)
        std_s = [len(npz_files)] + list(sample['std'].shape)
        text_features_s = [len(npz_files)] + list(sample['text_features'].shape)
        text_features_c_s = [len(npz_files)] + list(sample['text_features_c'].shape)
        if self.concat_text_fc: 
            text_features_c_s[-1] = text_features_c_s[-1] + text_features_s[-1]

        log.info(f'Loading {len(npz_files)} npz files from {npz_dir}')
        log.info(f'Loaded mean: {mean_s}.')
        log.info(f'Loaded std: {std_s}.')
        log.info(f'Loaded text features: {text_features_s}.')
        log.info(f'Loaded text features_c: {text_features_c_s}.') 

        # assert len(npz_files) == len(self.df_list), f'Number mismatch between npz files and tsv items: {len(npz_files)} vs {len(self.df_list)}. metadata_path: {metadata_path}'
        if len(npz_files) != len(self.df_list):
            log.warning(f'Number mismatch between npz files and tsv items: {len(npz_files)} vs {len(self.df_list)}. metadata_path: {metadata_path}')
        assert mean_s[1] == self.data_dim['latent_seq_len'], \
            f'{mean_s[1]} != {self.data_dim["latent_seq_len"]}'
        assert std_s[1] == self.data_dim['latent_seq_len'], \
            f'{std_s[1]} != {self.data_dim["latent_seq_len"]}'
        assert text_features_s[1] == self.data_dim['text_seq_len'], \
            f'{text_features_s[1]} != {self.data_dim["text_seq_len"]}'
        assert text_features_s[-1] == self.data_dim['text_dim'], \
            f'{text_features_s[-1]} != {self.data_dim["text_dim"]}'
        assert text_features_c_s[-1] == self.data_dim['text_c_dim'], \
            f'{text_features_c_s[-1]} != {self.data_dim["text_c_dim"]}'
    
        self.npz_dir = npz_dir
        if repa_npz_dir != None: 
            self.repa_npz_dir = repa_npz_dir
            sample = np.load(f'{repa_npz_dir}/0.npz')
            repa_npz_files = glob.glob(f"{repa_npz_dir}/*.npz")
            log.info(f'Loading {len(repa_npz_files)} npz representations from {repa_npz_dir}')
            es_s = [len(repa_npz_files)] + list(sample['es'].shape)
            if self.repa_version == 2: 
                es_s[1] = 65  # ad-hoc 8x downsampling for EAT 
            elif self.repa_version == 3: 
                es_s[1] = 1   # we only use cls token for alignment 
            else: 
                if self.exclude_cls: 
                    es_s[1] = es_s[1] - 1

            log.info(f'Loaded es: {es_s}')
            assert len(repa_npz_files) == len(npz_files), 'Number mismatch between repa npz files and latent npz files'
            assert es_s[1] == self.data_dim['repa_seq_len'], \
                f'{es_s[1]} != {self.data_dim["repa_seq_len"]}'
            assert es_s[-1] == self.data_dim['repa_seq_dim'], \
                f'{es_s[-1]} != {self.data_dim["repa_seq_dim"]}'
        else: 
            self.repa_npz_dir = None

    def compute_latent_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        # !TODO here we may consider load pre-computed latent mean & std
        raise NotImplementedError('Please manually compute latent stats outside. ')

    @staticmethod
    def read_metadata(metadata_path):
        if metadata_path is None:
            return None
        metadata_path = str(metadata_path)
        if metadata_path.endswith('.tsv'):
            return pd.read_csv(metadata_path, sep='\t').to_dict('records') # id, caption
        if metadata_path.endswith('.jsonl'):
            with open(metadata_path, encoding='utf8') as f:
                lines = f.read().splitlines()
            return [json.loads(line) for line in lines]
        raise NotImplementedError(f"Unkown metadata format: {metadata_path}")
    
    def sample_item(self, idx):
        npz_path = f'{self.npz_dir}/{idx}.npz'
        np_data = np.load(npz_path)
        text_features = torch.from_numpy(np_data['text_features'])
        text_features_c = torch.from_numpy(np_data['text_features_c'])
        if self.concat_text_fc: 
            text_features_c = torch.cat([text_features.mean(dim=-2),
                                         text_features_c], dim=-1)   # [b, d+d_c]

        out_dict = {
            'id': str(self.df_list[idx]['id']),
            'a_mean': torch.from_numpy(np_data['mean']), 
            'a_std': torch.from_numpy(np_data['std']), 
            'text_features': text_features, 
            'text_features_c': text_features_c,
            'caption': self.df_list[idx]['caption'],
        }
        
        # 添加condition返回
        # breakpoint()
        out_dict.update(extract_all_conditions(self, self.df_list[idx]))
        if 'insert' in out_dict:
            out_dict['target'] = out_dict['insert']['target']
            del out_dict['a_mean'], out_dict['a_std'], out_dict['insert']['target']
            out_dict['text_features'] = self.empty_string_feat
            out_dict['text_features_c'] = self.empty_string_feat_c
        
        # NOTE: 实际上对于remove和insert来说应该要把caption置为空的
        if 'remove' in out_dict or 'insert' in out_dict:
            out_dict['caption'] = ''
        # import torchaudio
        # torchaudio.save('mix.wav', out_dict['reference']['audio'], 44100)
        # _wav, _sr = torchaudio.load(self.df_list[idx]['path'])
        # torchaudio.save('tar.wav', _wav, _sr)
        # print(out_dict['remove'])
        # breakpoint()
        
        if self.repa_npz_dir != None: 
            repa_npz_path = f'{self.repa_npz_dir}/{idx}.npz'
            repa_np_data = np.load(repa_npz_path)
            zs =  torch.from_numpy(repa_np_data['es'])   

            if self.repa_version == 1: 
                if self.exclude_cls: 
                    zs = zs[1:,:]
            if self.repa_version == 2: 
                z_cls = zs[0]  # (dim)
                # zs = zs[1:,:].view(64, 8, 768)  
                zs = F.avg_pool2d(zs[1:,:].unsqueeze(0), 
                                  kernel_size=(8, 1), 
                                  stride=(8, 1)).squeeze()  # (64, 768)
                zs = torch.cat((z_cls.unsqueeze(0), zs), dim=0)
            elif self.repa_version == 3:  # cls token
                zs = zs[0].unsqueeze(0)
                
            out_dict['zs'] = zs  #!TODO Here field is WRONG for eat features (should be zs)
        
        return out_dict

    def __getitem__(self, idx):
        while True:
            try:
                return self.sample_item(idx)
            except Exception as e:
                from traceback import print_exc
                print_exc()
                idx = random.randint(0, len(self)-1)

    def __len__(self):
        return len(self.ids)
    

if __name__ == '__main__': 

    from audio_controlnet.utils.dist_utils import info_if_rank_zero, local_rank, world_size
    import torch.distributed as distributed
    from datetime import timedelta
    from torch.utils.data.distributed import DistributedSampler


    def distributed_setup():
        distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
        return local_rank, world_size

    distributed_setup()

    metadata_path = '/hpc_stor03/sjtu_home/xiquan.li/TTA/MMAudio/training/audiocaps/train-memmap-t5-clap.tsv'

    data_dim = {'latent_seq_len': 312, 
                'text_seq_len': 77,
                'text_dim': 1024, 
                'text_c_dim': 512}

    dataset = ExtractedAudio(metadata_path=metadata_path,
                                    npz_dir=npz_dir,
                                    data_dim=data_dim)
    loader = DataLoader(dataset,
                        16,
                        num_workers=8,
                        persistent_workers=8,
                        pin_memory=False)
    train_sampler = DistributedSampler(dataset, rank=local_rank, shuffle=True)


    for b in loader: 
        print(b['a_mean'].shape)
        break