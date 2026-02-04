import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
import torch
import torchaudio
import shutil
from audio_controlnet.eval_utils import (ModelConfig, generate_fm, setup_eval_logging, get_model_config)
from audio_controlnet.model.flow_matching import FlowMatching
from audio_controlnet.model.mean_flow import MeanFlow
from audio_controlnet.model.networks import FluxAudio, build_text2audio_model
from audio_controlnet.utils.runner_utils import extract_control_kwargs_from_data, extract_model_kwargs
from audio_controlnet.model.utils.features_utils import FeaturesUtils
from audio_controlnet.data.extract_cond import extract_condition, extract_all_conditions, load_audio_mono, expand_to, read_and_resample, savgol_filter, extract_pitch_contour
from audio_controlnet.data.extracted_audio import ExtractedAudio
from audio_controlnet.infer.utils import get_local_model_dir, get_model_config_and_path, load_weights_auto, merge_weights, omegaconf_resolve, omegaconf_load
from omegaconf import OmegaConf
import librosa
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
log = logging.getLogger()
from types import SimpleNamespace

def load_args(**kwargs):
    # default arguments for inference
    defaults = dict(
        variant='fluxaudio_m_full',
        negative_prompt='',
        duration=10.0,
        cfg_strength=4.5,
        num_steps=25,
        full_precision=True,
        model_path=None,
        model_dir=None,
        model_config=None,
        encoder_name='t5_clap',
        use_rope=True,
        text_c_dim=512,
        debug=False,
        use_meanflow=False,
        from_data_to_noise=True,
    )

    defaults.update(kwargs)

    return SimpleNamespace(**defaults)


@torch.no_grad()
def load_models(args, device):
    setup_eval_logging()
    
    if args.debug: 
        import debugpy
        debugpy.listen(6666) 
        print("Waiting for debugger attach (rank 0)...")
        debugpy.wait_for_client()  
    
    
    # load training config
    model_cfg = omegaconf_load(args.model_config)
    base_model_dir = get_local_model_dir(model_cfg.base_model) if 'base_model' in model_cfg else args.model_dir
    model_cfg = omegaconf_resolve(model_cfg, env={'base_model_dir': base_model_dir})
    
    # For ControlNets
    if 'base_model' in model_cfg:
        base_model_config, base_model_path = get_model_config_and_path(base_model_dir)
        base_model_config = omegaconf_resolve(OmegaConf.load(base_model_config), env={'base_model_dir': base_model_dir})
        model_cfg = OmegaConf.merge(base_model_config, model_cfg)
        model: ModelConfig = get_model_config(model_cfg.model_config, base_model_dir)
    # For base model (i.e. FluxAudio)
    else:
        model: ModelConfig = get_model_config(model_cfg.model_config, args.model_dir)
    
    seq_cfg = model.seq_cfg  

    num_steps: int = args.num_steps
    duration: float = args.duration

    if device == 'auto':
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            log.warning('CUDA/MPS are not available, running on CPU')
            
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    # load a pretrained model
    net: FluxAudio = build_text2audio_model(model.model_name, 
                                    use_rope=args.use_rope, 
                                    text_c_dim=args.text_c_dim,
                                    **extract_model_kwargs(model_cfg),
                                    ).to(device, dtype).eval() 
    if str(args.model_path) != 'None':
        weights = load_weights_auto(args.model_path, device=device)
        if 'base_model' in model_cfg:
            base_weights = load_weights_auto(base_model_path, device=device) 
            weights = merge_weights([base_weights, weights])
        
        net.load_weights(weights)
    log.info(f'Loaded weights from {args.model_path}')

    # misc setup
    # rng = torch.Generator(device=device)
    # rng.manual_seed(seed)
    if args.use_meanflow:
        mf = MeanFlow(steps=num_steps)
    else:
        fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps, from_data_to_noise=args.from_data_to_noise)
    
    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  enable_conditions=True,
                                  encoder_name=args.encoder_name, 
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  clap_ckpt_path=model.clap_ckpt_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device, dtype).eval()

    seq_cfg.duration = duration
    # net.update_seq_lengths(seq_cfg.latent_seq_len)
    
    return feature_utils, net, fm, model_cfg, seq_cfg

@dataclass
class AudioOutput:
    audio: torch.Tensor
    sample_rate: int

@torch.no_grad()
def run_infer(models, args, m):
    feature_utils, net, fm, model_cfg, seq_cfg = models
    negative_prompt: str = args.negative_prompt
    cfg_strength: float = args.cfg_strength
    device = next(net.parameters()).device
    
    
    def collate(b):
        if isinstance(b[0], torch.Tensor):
            return torch.stack(b).to(device)
        else:
            return b
        
    prompt = m.get('caption', '')
    extra_kwargs = {}
    if 'control' in m:
        if m['control'] is not None:
            ctl = m['control']
            ctl = {k:collate([v]) for k,v in ctl.items()}
            controls = [ctl]
            extra_kwargs['control'] = controls[0]
        else:
            del m['control']
        
    
    log.info(f'Prompt: {prompt}')
    
    audios = generate_fm([prompt],
                        negative_text=[negative_prompt],
                        feature_utils=feature_utils,
                        net=net,
                        fm=fm,
                        rng=None,
                        cfg_strength=cfg_strength,
                        # control=controls[0],
                        **extra_kwargs,
                        )
    
    audio = audios.float().cpu()[0]
    return AudioOutput(
        audio=audio,
        sample_rate=seq_cfg.sampling_rate,
    )


@dataclass
class AudioInput:
    audio: torch.Tensor
    sr: int
    
@dataclass
class ConditionInput:
    value: torch.Tensor

@dataclass
class LoudnessCondition(ConditionInput):
    value: torch.Tensor
    
@dataclass
class PitchCondition(ConditionInput):
    value: torch.Tensor
    f0: np.ndarray

@dataclass
class EventsCondition(ConditionInput):
    value: torch.Tensor

class Runner:
    def __init__(self, args, device='auto') -> None:
        models = load_models(args, device=device)
        self.models, self.args = models, args
        self.feature_utils, self.net, self.fm, self.model_cfg, self.seq_cfg = models
        self.default_control_types = deepcopy(self.net.control_types if hasattr(self.net, 'control_types') else [])
        
    def infer(self, caption='', control=None, control_types=None, **meta):
        self.net.control_types = self.default_control_types if control_types is None else control_types # specified or default
        self.net.control_types = [ct for ct in self.net.control_types if ct in control] # use only the control types in the input
        if control is not None:
            for k in control:
                if isinstance(control[k], ConditionInput):
                    control[k] = control[k].value
        return run_infer(self.models, self.args, dict(caption=caption, control=control, **meta))
    
    @property
    def sample_rate(self):
        return self.models[-1].sampling_rate

    def prepare_audio(self, path):
        audio, sr = load_audio_mono(path, self.sample_rate)
        return AudioInput(audio, sr)
    
    def prepare_loudness(self, path):
        seq_len = self.model_cfg.conditioners_config.loudness.seq_len
        target_rate = self.model_cfg.conditioners_config.loudness.target_rate
        
        y = read_and_resample(path, self.sample_rate)
            
        # Set hop_length to make the RMS sampling frequency equal to target_rate
        hop_length = int(self.sample_rate / target_rate)  
        frame_length = 4096 

        # calculate RMS
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # calculate dB
        min_rms = 1e-6
        rms = np.maximum(rms, min_rms)
        dB = 20 * np.log10(rms / np.max(rms))

        # savgol filter
        smooth_savgol = savgol_filter(dB, window_length=11, polyorder=3)

        return LoudnessCondition(expand_to(torch.from_numpy(smooth_savgol), len=seq_len))

    def prepare_pitch(self, path):
        seq_len = self.model_cfg.conditioners_config.pitch.seq_len
        target_rate = self.model_cfg.conditioners_config.pitch.target_rate
        
        
        i_pitch, f0 = extract_pitch_contour(path, target_rate=target_rate, seq_len=seq_len)
        
        return PitchCondition(torch.from_numpy(i_pitch).long(), f0)

    def plot_to_image(self, data, fpath, tag='Value', dpi=160, xlabel='Time(sec)', ylabel=None, target_rate=43.0, line_color=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        
        if isinstance(data, PitchCondition):
            data = data.f0
        elif isinstance(data, LoudnessCondition):
            data = data.value

        if "torch" in str(type(data)):
            data = data.detach().cpu().numpy()

        data = np.array(data)

        plt.figure(figsize=(8, 4), dpi=dpi)

        if data.ndim == 2:
            for i in range(data.shape[1]):
                plt.plot(data[:, i], label=f"{tag}_{i}", color=line_color)
            plt.legend()
        else:
            plt.plot(data, label=tag, color=line_color)
            plt.legend()

        plt.title(tag)
        plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # xtrcik div 25
        formatter = FuncFormatter(lambda x, _: round(x / target_rate, 1))
        plt.gca().xaxis.set_major_formatter(formatter)

        # save file
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()  
        
        
class AudioControlNet(Runner):
    def __init__(self, model_config, model_path, model_dir, device) -> None:
        args = load_args(model_config=model_config, model_path=model_path, model_dir=model_dir)
        super().__init__(args, device=device)
    
    @classmethod
    def from_pretrained(cls, model_name, device='auto'):
        model_dir = get_local_model_dir(model_name)
        model_config, model_path = get_model_config_and_path(model_dir)
        
        return cls(model_config, model_path, model_dir, device=device)
        
    @classmethod
    def from_multi_controlnets(cls, model_name_list, device='auto'):
        model_dir_list = [get_local_model_dir(model_name) for model_name in model_name_list]
        model_config_list, model_path_list = [], [] 
        for model_dir in model_dir_list:
            model_config, model_path = get_model_config_and_path(model_dir)
            model_config_list.append(model_config)
            model_path_list.append(model_path)
        return cls(model_config_list, model_path_list, model_dir_list, device=device)
        
        