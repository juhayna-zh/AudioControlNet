from copy import deepcopy
import torch

def extract_control_kwargs_from_data(data, cfg):
    if cfg.get('control_types', None) is None:
        return dict()
    control = {}
    for name in cfg.control_types:
        control[name] = data[name]
    return dict(control=control)

MODEL_KWARGS_KEYS = set([
    'latent_dim', 'text_dim', 'hidden_dim', 'depth', 'fused_depth', 'num_heads', 'latent_seq_len',
    'control_types', 'base_model_checkpoint', 'conditioners_config', 'controlnets_config', 'use_lora', 'lora_rank', 
])

def extract_model_kwargs(cfg):
    new_kwargs = {}
    for k, v in cfg.items():
        if v is not None and k in MODEL_KWARGS_KEYS:
            new_kwargs[k] = deepcopy(v)
    return new_kwargs

def extract_mask_unmask_loss_coef(segments, target_tensor, target_rate, mask_ratio=1.0, unmask_ratio=1.0):
    bsz, seq_len, feat_dim = target_tensor.shape
    coef_tensor = torch.ones_like(target_tensor) * unmask_ratio
    for i in range(bsz):
        for st, et in segments[i]:
            s_idx = round(st * target_rate)
            e_idx = round(et * target_rate)
            coef_tensor[i, s_idx:e_idx, :] = mask_ratio
    return coef_tensor
    
    