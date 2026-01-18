from copy import deepcopy
import torch

def extract_control_kwargs_from_data(data, cfg):
    if cfg.get('control_types', None) is None:
        return dict()
    control = {}
    for name in cfg.control_types:
        control[name] = data[name]
    return dict(control=control)

def safe_kwargs(**kwargs):
    new_kwargs = deepcopy(kwargs)
    for k, v in kwargs.items():
        if v is None:
            del new_kwargs[k]
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
    
    