import torch
import torch.nn as nn
import typing as tp
import os
import numpy as np
from typing import Literal
import logging
from functools import lru_cache

log = logging.getLogger()

EventType = tp.Dict[str, tp.List[tp.List[float]]]

class Conditioner(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: tp.Any, device=None) -> torch.Tensor:
        raise NotImplementedError()


def norm_name(label):
    return label.replace('/', '_')

class EventsTextConditioner(Conditioner):
    def __init__(self, embed_dirs: tp.List[os.PathLike] = [], output_dim: int = 20, seq_len = 312, target_rate = 31.2, \
            fuse_method:Literal['add', 'qformer', 'flatten'] = 'add', qformer_config = None, flatten_config = None, embed_dims = None, clap_path = './weights/music_speech_audioset_epoch_15_esc_89.98.pt'):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dirs = embed_dirs
        
        self.embed_dicts = []
        self.embed_dims = []
        self.seq_len = seq_len
        self.target_rate = target_rate
        self.fuse_method = fuse_method
        
        # 预先读取embedding
        for d in self.embed_dirs:
            embed_dict = {}
            for fname in os.listdir(d):
                label, _ext = os.path.splitext(fname)
                embed = np.load(os.path.join(d, fname))
                embed_dict[label] = embed
            self.embed_dims.append(embed.shape[-1])
            self.embed_dicts.append(embed_dict)
        
        if embed_dims is not None:
            self.embed_dims = embed_dims
        self.clap = None
        self.clap_path = clap_path 
        
        if self.fuse_method == 'add':
            # 线性映射到output_dim
            self.linear = nn.Linear(sum(self.embed_dims), output_dim)
        elif self.fuse_method == 'qformer':
            from .qformer import EventQFormerWrapper
            from omegaconf import OmegaConf
            self.qformer = EventQFormerWrapper(OmegaConf.create(qformer_config))
        elif self.fuse_method == 'flatten':
            self.flatten_config = flatten_config
            self.linear = nn.Linear(sum(self.embed_dims), output_dim)
            if flatten_config['use_sep_token']:
                self.sep_emb = nn.Parameter(torch.randn(output_dim))
            if flatten_config['use_pad_token']:
                self.pad_emb = nn.Parameter(torch.randn(output_dim))
        else:
            raise NotImplementedError(f'Unknown fuse_method: {self.fuse_method}')
        
        
    def lookup_embed(self, label):
        label_norm = norm_name(label)
        if len(self.embed_dicts) > 0 and all([label_norm in embed_dict for embed_dict in self.embed_dicts]):
            embeds = [torch.from_numpy(embed_dict[norm_name(label)]) for embed_dict in self.embed_dicts]
            return torch.cat(embeds)
        if self.clap is None:
            import laion_clap
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
            model.load_ckpt(self.clap_path, verbose=False) 
            self.clap = NonTrainableWrapper(model)
        text_data = [label] 
        text_embed = self.clap.model.get_text_embedding(text_data)[0]
        embeds = [torch.from_numpy(text_embed)]
        return torch.cat(embeds)

    def forward(self, x: tp.List[EventType], device=None) -> torch.Tensor:
        ret = []
        
        seq_len = self.seq_len
        target_rate = self.target_rate

        # 处理每个事件，将其时间步对应的张量直接加在result_tensor上
        if self.fuse_method == 'add':
            for events in x:
                result_tensor = torch.zeros((self.output_dim, seq_len), ).float().to(device)
                for label, time_list in events.items(): 
                    this_embed = self.lookup_embed(label).to(device)
                    this_embed = self.linear(this_embed)
                    for st, et in time_list:
                        start = round(st*target_rate)
                        end = min(round(et*target_rate), seq_len)
                        result_tensor[:, start:end] += this_embed.unsqueeze(1).repeat(1, end - start)
                
                ret.append(result_tensor)

            ret = torch.stack(ret).permute(0, 2, 1) # [batch, hidden_dim, seq_len] -> [batch, seq_len, hidden_dim]
            
        elif self.fuse_method == 'qformer':
            num_events = [len(events) for events in x]
            N = max(num_events)
            ret = []
            for events in x:
                result_tensor = torch.zeros((N, sum(self.embed_dims), seq_len)).float().to(device)
                for i, (label, time_list) in enumerate(events.items()): 
                    this_embed = self.lookup_embed(label).to(device)
                    for st, et in time_list:
                        start = round(st*target_rate)
                        end = min(round(et*target_rate), seq_len)
                        result_tensor[i, :, start:end] = this_embed.unsqueeze(1).repeat(1, end - start)
                ret.append(result_tensor)
            ret = torch.stack(ret)
            ret = self.qformer(ret, num_events) # -> [B, T, output_dim]
            
        elif self.fuse_method == 'flatten':
            # breakpoint()
            batch_result = []

            for events in x:
                result_seq = [[] for _ in range(self.seq_len)]

                # 遍历 label（字符串），逐个 lookup
                for label, time_list in events.items():
                    this_embed = self.linear(self.lookup_embed(label).to(device))
                    for st, et in time_list:
                        start = round(st * target_rate)
                        end = round(et * target_rate)
                        # torch.arange 代替 python for
                        t_idx = torch.arange(start, end, device=device)
                        for t in t_idx.tolist():
                            result_seq[t].append(this_embed)

                # 构建 result_tensor
                result_tensor = []
                for sub_seq in result_seq:
                    result_tensor.extend(sub_seq)
                    if self.flatten_config['use_pad_token']:
                        pad_count = len(events) - len(sub_seq)
                        if pad_count > 0:
                            result_tensor.extend([self.pad_emb] * pad_count)
                    if self.flatten_config['use_sep_token']:
                        result_tensor.append(self.sep_emb)

                if len(result_tensor) == 0:
                    result_tensor = torch.zeros(1, self.linear.out_features, device=device)
                else:
                    result_tensor = torch.stack(result_tensor)

                batch_result.append(result_tensor)

            # padding
            max_len = max(r.size(0) for r in batch_result)
            hidden_dim = batch_result[0].size(-1)
            ret = torch.zeros(len(batch_result), max_len, hidden_dim, device=device, )
            for i, r in enumerate(batch_result):
                ret[i, :r.size(0)] = r
            
        return ret.contiguous() # [batch, seq_len, hidden_dim]


class EditConditioner(EventsTextConditioner):
    def __init__(self, *, mode, tod_vae_ckpt, bigvgan_vocoder_ckpt=None, vae_latent_dim=40, use_joint_conditioner=False, **kwargs):
        assert use_joint_conditioner, f"Support joint conditioner only but got use_joint_conditioner={use_joint_conditioner}"
        super().__init__(**kwargs)
        
        tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                mode=mode).eval()
        self.tod = NonTrainableWrapper(tod)
        mel_converter = get_mel_converter(mode).eval()
        self.mel_converter = NonTrainableWrapper(mel_converter)

        self.sample_rate = {'16k': 16000, '44k': 44100}[mode]
        
        self.final_linear = nn.Linear(self.output_dim+vae_latent_dim, self.output_dim)

    def forward(self, inputs: tp.Dict[str, tp.Any], device=None) -> torch.Tensor:
        
        # 获取events的embedding
        events = [x['events'] for x in inputs]
        events_emb = super().forward(events, device) # [batch, seq_len, hidden_dim]
        
        # 获取reference的vae latent
        self.tod.model.to(device)
        self.mel_converter.model.to(device)
        
        # 对输入的音频padding到需要的长度
        target_len = int(self.seq_len / self.target_rate * self.sample_rate)
        # breakpoint()
        batch_waveforms = []
        for datum in inputs:
            audio = datum['reference']['audio'].to(device)

            # 1. padding / trim 到目标长度
            if audio.shape[-1] < target_len:
                pad_len = target_len - audio.shape[-1]
                audio = F.pad(audio, (0, pad_len))
            else:
                audio = audio[:, :target_len]  # 假设 audio shape [1, T]

            batch_waveforms.append(audio.clone())

        x = torch.stack(batch_waveforms, dim=0)  # [B, 1, T]
        
        # 获取audio的vae latent
        mel = self.mel_converter.model(x.squeeze(dim=1))
        dist = self.tod.model.encode(mel)
        if 'a_randn' in inputs[0]:
            a_randn = torch.stack([inp['a_randn'] for inp in inputs])
            latent = (dist.mean + dist.std * a_randn.permute(0,2,1)).detach()
        else:
            latent = dist.sample().detach() # [B, output_dim, seq_len]
        latent = latent.permute(0, 2, 1) # [B, seq_len, output_dim]
            
        # 进行最后的linear
        return self.final_linear(torch.cat([events_emb, latent], dim=-1))

    @torch.no_grad()
    def get_vae_mean_and_std(self, wavs):
        device = wavs.device
        self.tod.model.to(device)
        self.mel_converter.model.to(device)
        
        mel = self.mel_converter.model(wavs)
        dist = self.tod.model.encode(mel)
        
        return dist.mean.permute(0, 2, 1).detach(), dist.std.permute(0, 2, 1).detach()

class LoudnessConditioner(Conditioner):
    def __init__(self, output_dim: int = 20, seq_len = 312, target_rate = 31.2,):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.target_rate = target_rate

    def forward(self, x: torch.Tensor, device=None) -> torch.Tensor:
        # 得到形状为 [B, 1, Seq] 的张量
        x = x.unsqueeze(1)  # [B, 1, Seq]
        
        # 将张量沿着output_dim维度重复，得到形状为 [B, output_dim, Seq]
        x = x.repeat(1, self.output_dim, 1)  # [B, output_dim, Seq]
            
        return x.permute(0, 2, 1).to(device).contiguous() # [batch, seq_len, hidden_dim]

class PitchCWTQConditioner(Conditioner):
    def __init__(self, output_dim=40, n_bins=256, H=31, emb_dim=64, seq_len = 312, target_rate = 31.2,):
        super().__init__()
        self.H = H
        self.emb_dim = emb_dim
        self.output_dim = output_dim

        # 每个scale一套embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_bins, emb_dim) for _ in range(H)
        ])

        # 最终投影到统一维度 D
        self.proj = nn.Linear(H * emb_dim, output_dim)

    def forward(self, i_pitch, device=None):
        """
        Args:
            i_pitch: (B, L, H)，量化后的pitch，int类型，取值[0, n_bins-1]
        Returns:
            x: (B, L, out_dim)，可作为下游模型输入
        """
        i_pitch = i_pitch.to(device)
        B, L, H = i_pitch.shape
        assert H == self.H, f"input H={H}, but expect H={self.H}"

        # 对每个scale做embedding
        emb_list = []
        for h in range(H):
            emb = self.embeddings[h](i_pitch[:, :, h])  # (B, L, emb_dim)
            emb_list.append(emb)

        # 在最后一维拼接 → (B, L, H*emb_dim)
        x = torch.cat(emb_list, dim=-1)

        # 投影到 out_dim → (B, L, out_dim)
        x = self.proj(x)

        return x.to(device)

from collections import namedtuple
NonTrainableWrapper = namedtuple("NonTrainableWrapper", ["model"])

from audio_controlnet.ext.autoencoder import AutoEncoderModule
from audio_controlnet.ext.mel_converter import get_mel_converter
import torch.nn.functional as F
    
class InpaintConditioner(Conditioner):
    def __init__(self, mode, tod_vae_ckpt, bigvgan_vocoder_ckpt=None, output_dim=40, seq_len = 312, target_rate = 31.2, use_joint_conditioner=False, vae_latent_dim=None):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.target_rate = target_rate
        
        tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                mode=mode).eval()
        self.tod = NonTrainableWrapper(tod)
        mel_converter = get_mel_converter(mode).eval()
        self.mel_converter = NonTrainableWrapper(mel_converter)

        self.sample_rate = {'16k': 16000, '44k': 44100}[mode]
        
        self.use_joint_conditioner = use_joint_conditioner
        if self.use_joint_conditioner:
            self.final_linear = nn.Linear(vae_latent_dim + 1, self.output_dim)
        else:
            self.mask_emb = nn.Parameter(torch.zeros(output_dim))

    def forward(self, inputs, device=None):
        """
        Args:
            inputs: list[dict], 每个元素都是字典: {"audio": Tensor<'mono channel audio'>, "segments": list[tuple] }
        Returns:
            x: (B, L, out_dim)，可作为下游模型输入
        """
        self.tod.model.to(device)
        self.mel_converter.model.to(device)
        
        # 对输入的音频padding到需要的长度，并对其segment部分进行静默处理
        target_len = int(self.seq_len / self.target_rate * self.sample_rate)
        # breakpoint()
        batch_waveforms = []
        for datum in inputs:
            audio = datum['audio'].to(device)

            # 1. padding / trim 到目标长度
            if audio.shape[-1] < target_len:
                pad_len = target_len - audio.shape[-1]
                audio = F.pad(audio, (0, pad_len))
            else:
                audio = audio[:, :target_len]  # 假设 audio shape [1, T]

            # 2. 对 segments 进行静默处理（mask wave）
            mask_audio = audio.clone()
            # breakpoint()
            for st, et in datum['segments']:
                s_idx = round(st * self.sample_rate)
                e_idx = round(et * self.sample_rate)
                mask_audio[:, s_idx:e_idx] = 0.0
            batch_waveforms.append(mask_audio)

        x = torch.stack(batch_waveforms, dim=0)  # [B, 1, T]
        
        # 获取audio的vae latent
        mel = self.mel_converter.model(x.squeeze(dim=1))
        dist = self.tod.model.encode(mel)
        if 'a_randn' in inputs[0]:
            a_randn = torch.stack([inp['a_randn'] for inp in inputs])
            latent = (dist.mean + dist.std * a_randn.permute(0,2,1)).detach()
        else:
            latent = dist.sample().detach() # [B, output_dim, seq_len]
        
        if self.use_joint_conditioner:
            # 将segments作为0-1 mask拼到latent上
            B, _vae_dim, _seq_len = latent.shape
            segments_mask = torch.zeros((B, 1, _seq_len), dtype=latent.dtype).to(latent.device)
            for i, datum in enumerate(inputs):
                for st, et in datum['segments']: # st和et都是以秒为单位
                    # 将时间映射到 latent 时间轴
                    s_idx = round(st * self.target_rate)  # 简单线性映射
                    e_idx = min(round(et * self.target_rate), target_len)
                    segments_mask[i, :, s_idx:e_idx] = 1
            latent = self.final_linear(torch.cat([latent, segments_mask], dim=1).permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # 对vae latent的segment部分置为mask_emb
            mask = self.mask_emb.unsqueeze(-1)  # [D, 1]
            for i, datum in enumerate(inputs):
                for st, et in datum['segments']: # st和et都是以秒为单位
                    # 将时间映射到 latent 时间轴
                    s_idx = round(st * self.target_rate)  # 简单线性映射
                    e_idx = min(round(et * self.target_rate), target_len)
                    # latent[i, :, s_idx:e_idx] = self.mask_emb.unsqueeze(-1)
                    mask_region = torch.zeros_like(latent[i, :, s_idx:e_idx])
                    mask_region = mask_region + mask  # this line creates a differentiable op
                    latent[i, :, s_idx:e_idx] = latent[i, :, s_idx:e_idx] * 0 + mask_region

        return latent.permute(0, 2, 1) # [B, seq_len, output_dim]


class ReferenceConditioner(Conditioner):
    def __init__(self, mode, tod_vae_ckpt, bigvgan_vocoder_ckpt=None, output_dim=40, seq_len = 312, target_rate = 31.2,):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.target_rate = target_rate
        
        tod = AutoEncoderModule(vae_ckpt_path=tod_vae_ckpt,
                                vocoder_ckpt_path=bigvgan_vocoder_ckpt,
                                mode=mode).eval()
        self.tod = NonTrainableWrapper(tod)
        mel_converter = get_mel_converter(mode).eval()
        self.mel_converter = NonTrainableWrapper(mel_converter)

        self.sample_rate = {'16k': 16000, '44k': 44100}[mode]

    def forward(self, inputs, device=None):
        """
        Args:
            inputs: list[dict], 每个元素都是字典: {"audio": Tensor<'mono channel audio'>, "segments": list[tuple] }
        Returns:
            x: (B, L, out_dim)，可作为下游模型输入
        """
        self.tod.model.to(device)
        self.mel_converter.model.to(device)
        
        # 对输入的音频padding到需要的长度，并对其segment部分进行静默处理
        target_len = int(self.seq_len / self.target_rate * self.sample_rate)
        # breakpoint()
        if 'audio_vae' not in inputs[0]:
            batch_waveforms = []
            for datum in inputs:
                audio = datum['audio'].to(device)

                # 1. padding / trim 到目标长度
                if audio.shape[-1] < target_len:
                    pad_len = target_len - audio.shape[-1]
                    audio = F.pad(audio, (0, pad_len))
                else:
                    audio = audio[:, :target_len]  # 假设 audio shape [1, T]

                batch_waveforms.append(audio.clone())

            x = torch.stack(batch_waveforms, dim=0)  # [B, 1, T]
            
            # 获取audio的vae latent
            mel = self.mel_converter.model(x.squeeze(dim=1))
            dist = self.tod.model.encode(mel)
            # TODO: Add a_randn support
            # dist.mean.shape: torch.Size([24, 40, 430]) ; dist.std.shape: torch.Size([24, 40, 430])
            if 'a_randn' in inputs[0]:
                a_randn = torch.stack([inp['a_randn'] for inp in inputs])
                latent = (dist.mean + dist.std * a_randn.permute(0,2,1)).detach()
            else:
                latent = dist.sample().detach() # [B, output_dim, seq_len]
        else:
            # breakpoint()
            latent = torch.stack([inp['audio_vae'] for inp in inputs]).permute(0, 2, 1)

        return latent.permute(0, 2, 1)



def create_conditioner_from_config(name, cfg):
    cfg_type = cfg['type']
    kwargs = {**cfg}
    del kwargs['type']
    
    if cfg_type in ('events-text', 'events_text'):
        return EventsTextConditioner(**kwargs)
    elif cfg_type == 'loudness':
        return LoudnessConditioner(**kwargs)
    elif cfg_type == 'pitch':
        return PitchCWTQConditioner(**kwargs)
    elif cfg_type == 'inpaint':
        return InpaintConditioner(**kwargs)
    elif cfg_type == 'reference':
        return ReferenceConditioner(**kwargs)
    elif cfg_type in ('remove', 'insert'):
        if kwargs.get("use_joint_conditioner", False):
            return EditConditioner(**kwargs)
        else:
            return EventsTextConditioner(**kwargs)
    else:
        raise NotImplementedError(f"No implementation for condtioner `{name}` with cfg = {cfg}")