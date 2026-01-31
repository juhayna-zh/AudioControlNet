import dataclasses
import logging
from pathlib import Path
from typing import Optional
import os.path as op

import numpy as np
import torch
from colorlog import ColoredFormatter
from PIL import Image
from torchvision.transforms import v2

from audio_controlnet.data.av_utils import ImageInfo, VideoInfo, read_frames, reencode_with_audio
from audio_controlnet.model.flow_matching import FlowMatching
from audio_controlnet.model.mean_flow import MeanFlow
from audio_controlnet.model.networks import MeanAudio, FluxAudio
from audio_controlnet.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from audio_controlnet.model.utils.features_utils import FeaturesUtils
from audio_controlnet.utils.download_utils import download_model_if_needed

log = logging.getLogger()


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str
    clap_ckpt_path: Optional[Path] = None

    @property
    def seq_cfg(self) -> SequenceConfig:
        if self.mode == '16k':
            return CONFIG_16K  # get sequence config when calling cfg.seq_cfgs
        elif self.mode == '44k':
            return CONFIG_44K

    def download_if_needed(self):
        raise NotImplementedError("Downloading models is not supported")
        download_model_if_needed(self.model_path)
        download_model_if_needed(self.vae_path)
        if self.bigvgan_16k_path is not None:
            download_model_if_needed(self.bigvgan_16k_path)


fluxaudio_fm = ModelConfig(model_name='fluxaudio_fm', 
                           model_path=Path('./weights/fluxaudio_fm.pth'),
                           vae_path=Path('./weights/v1-16.pth'),
                           bigvgan_16k_path=Path('./weights/best_netG.pt'),
                           mode='16k')
fluxaudio_m_full = ModelConfig(model_name='fluxaudio_m_full', 
                                model_path=Path('./weights/fluxaudio_m_full.pth'), 
                                vae_path=Path('./weights/v1-44.pth'),
                                bigvgan_16k_path=None,
                                mode='44k')
meanaudio_mf = ModelConfig(model_name='meanaudio_mf', 
                           model_path=Path('./weights/meanaudio_mf.pth'),
                           vae_path=Path('./weights/v1-16.pth'),
                           bigvgan_16k_path=Path('./weights/best_netG.pt'),
                           mode='16k')

all_model_cfg: dict[str, ModelConfig] = {
    'fluxaudio_fm': fluxaudio_fm, 
    'meanaudio_mf': meanaudio_mf, 
    'fluxaudio_m_full': fluxaudio_m_full,
}

def get_model_config(cfg, model_dir):
    model_dir = Path(model_dir)
    return ModelConfig(
        model_name = cfg.model_name,
        model_path = model_dir/cfg.model_path,
        vae_path = model_dir/cfg.vae_path if cfg.vae_path else None,
        bigvgan_16k_path = model_dir/cfg.bigvgan_16k_path if cfg.bigvgan_16k_path else None,
        mode = cfg.mode,
        clap_ckpt_path = model_dir/cfg.clap_ckpt_path if cfg.clap_ckpt_path else None,
    )

def generate_fm(
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: FluxAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    **kwargs
) -> torch.Tensor:
    # generate audio with vanilla flow matching

    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)

    if text is not None:
        text_features, text_features_c = feature_utils.encode_text(text)
    else:
        text_features, text_features_c = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(text_features, text_features_c, **kwargs)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def generate_mf(
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MeanAudio,
    mf: MeanFlow,
    rng: torch.Generator,
    cfg_strength: float,
) -> torch.Tensor:
    # generate audio with mean flow
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)

    if text is not None:
        text_features, text_features_c = feature_utils.encode_text(text)
    else:
        text_features, text_features_c = net.get_empty_string_sequence(bs)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(text_features, text_features_c)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, r, x: net.ode_wrapper(t, r, x, preprocessed_conditions, empty_conditions,
                                                      cfg_strength)
    x1 = mf.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level) # set up root logger <=> logging.getLogger().setLevel(log_level) 
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()  # to Console
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger() 
    log.setLevel(log_level)
    log.addHandler(stream)