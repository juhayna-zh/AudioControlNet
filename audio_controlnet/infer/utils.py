
        
from pathlib import Path
from huggingface_hub import snapshot_download
import os
import os.path as op
import torch
from typing import Iterable, Dict, Union, List
from collections import OrderedDict

def get_local_model_dir(model_id_or_path: str) -> str:
    """
    Supports:
    - local dir path
    - namespace/model_name
    """
    p = Path(model_id_or_path)

    # local path
    if p.exists():
        return str(p.resolve())

    # view as HF repo_id
    return snapshot_download(repo_id=model_id_or_path)

def get_model_config_and_path(model_dir):
    model_config = None
    
    for config_filename in ['config.json', 'config.yaml']:
        if op.exists(op.join(model_dir, config_filename)):
            model_config = op.join(model_dir, config_filename)
            break
    
    assert model_config is not None, f"Model config not found in {model_dir}."
    
    model_path = None
    for model_filename in ['model.safetensors', 'pytorch_model.bin', 'model.ckpt']:
        if op.exists(op.join(model_dir, model_filename)):
            model_path = op.join(model_dir, model_filename)
            break

    assert model_path is not None, f"Model file not found in {model_dir}."
    return model_config, model_path

def merge_weights(
    state_dicts: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    merged = OrderedDict()
    for sd in state_dicts:
        merged.update(sd)
    return merged

def load_weights_auto(
    ckpt_path: Union[str, Path, Iterable[Union[str, Path]]],
    device="cpu",
    weights_only=True,
) -> Dict[str, torch.Tensor]:
    """
    Automatically load model weights based on the suffix of the weight file.

    Supports:
      - .pt / .pth / .bin  -> torch.load
      - .safetensors      -> safetensors.torch.load_file
      - list of paths     -> loads each and merges using merge_weights

    Return:
      - state_dict (dict[str, Tensor])
    """
    def _load_single(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path), device=device)
        elif suffix in {".pt", ".pth", ".bin"}:
            return torch.load(path, map_location=device, weights_only=weights_only)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")

    if isinstance(ckpt_path, (str, Path)):
        return _load_single(ckpt_path)

    elif isinstance(ckpt_path, Iterable):
        state_dicts = [_load_single(p) for p in ckpt_path]
        return merge_weights(state_dicts)

    else:
        raise TypeError(f"ckpt_path must be a Path, str, or iterable of them, got {type(ckpt_path)}")


from omegaconf import OmegaConf, DictConfig, ListConfig

from contextlib import contextmanager

class ResolverManager:
    path_dict = {}
    
    @classmethod
    def path(cls, key):
        return cls.path_dict.get(key, '.')
        
    @classmethod
    @contextmanager
    def setup(cls, **kwargs):
        cls.path_dict.update(kwargs)
        yield
        cls.path_dict.clear()
        
OmegaConf.register_new_resolver('path', ResolverManager.path)

def omegaconf_resolve(cfg, env={}):
    with ResolverManager.setup(**env):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)
    return cfg


def merge_first_level_multiple(*cfg_list):
    if not cfg_list:
        return OmegaConf.create({})

    merged = {}
    
    for cfg in cfg_list:
        cfg = OmegaConf.to_container(cfg, resolve=False) # We dont resolve it before we parse the required base_model_dir
        if cfg is None:
            continue
        for key in cfg.keys():
            val = cfg.get(key)
            if key in merged:
                existing = merged[key]
                if isinstance(existing, dict) or isinstance(existing, DictConfig):
                    for k, v in val.items():
                        merged[key][k] = v
                elif isinstance(existing, list) or isinstance(existing, ListConfig):
                    merged[key] = existing + val
                else:
                    assert existing == val, f"All configs must must use the same `{key}`!"
            else:
                merged[key] = val

    return OmegaConf.create(merged)

def omegaconf_load(cfg_path):
    if isinstance(cfg_path, List):
        cfg_list = []
        for cp in cfg_path:
            cfg_list.append(OmegaConf.load(cp))
        return merge_first_level_multiple(*cfg_list)
    else:
        return OmegaConf.load(cfg_path)