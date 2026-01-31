
        
from pathlib import Path
from huggingface_hub import snapshot_download
import os
import os.path as op
import torch
from typing import Iterable, Dict
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
    return snapshot_download(
        repo_id=model_id_or_path,
        local_files_only=True,
    )

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


def load_weights_auto(
    ckpt_path,
    device="cpu",
    weights_only=True,
):
    """
    Automatically load model weights based on the suffix of the weight file.

    Supports:
      - .pt / .pth / .bin  -> torch.load
      - .safetensors      -> safetensors.torch.load_file

    Return:
      - state_dict (dict[str, Tensor])
    """
    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    suffix = ckpt_path.suffix.lower()

    if suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(ckpt_path), device=device)

    elif suffix in {".pt", ".pth", ".bin"}:
        return torch.load(
            ckpt_path,
            map_location=device,
            weights_only=weights_only,
        )

    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")



def merge_weights(
    state_dicts: Iterable[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    merged = OrderedDict()
    for sd in state_dicts:
        merged.update(sd)
    return merged

from omegaconf import OmegaConf

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

