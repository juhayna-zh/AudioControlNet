import torch

from torchaudio.functional import resample
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt, savgol_filter

from torchaudio.functional import resample

from .utils import random_mask_segments

import librosa
import torchaudio
import numpy as np

import random

def expand_to(tensor, len, pad_elem=0):
    # 原始张量长度
    n = tensor.size(0)
    
    if n >= len:
        # 如果原始长度≥len，直接截取前len个元素
        return tensor[:len]
    else:
        # 计算需要重复的次数
        pad_times = len - n
        paded_part = torch.ones((pad_times,)) * pad_elem
        # 拼接原始张量和重复部分
        expanded_tensor = torch.cat([tensor, paded_part])
        return expanded_tensor
    
def load_audio_mono(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)  # waveform: [channels, T]
    
    # 如果不是单声道，取平均
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # 重采样到目标采样率
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    
    return waveform, sr
    
import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_pitch_contour(
    audio_path: str,
    sr: int = 16000,
    fmin: str = "C2",
    fmax: str = "C7",
    n_bins: int = 256,
    wavelet_widths: np.ndarray = None,
    target_rate: float = None,  # 帧率 Hz
    seq_len: int = None,        # 期望帧数
    show_plot: bool = False, 
):
    """
    提取论文中描述的 pitch contour，并支持帧率重采样和序列长度调整。
    
    Args:
        audio_path: 音频路径
        sr: 采样率
        fmin, fmax: 基频搜索范围
        n_bins: 量化 bins 数量（默认 256）
        wavelet_widths: 小波尺度范围 (默认 [1..32])
        target_rate: float, 目标帧率 (Hz)，None表示不重采样
        seq_len: int, 目标序列长度
        show_plot: 是否显示可视化 heatmap
    
    Returns:
        i_pitch: 量化后的 pitch 序列 (L × H)
        f0: 原始基频轨迹 (Hz)，长度 L
        times: 时间轴 (s)，长度 L
    """
    # -------------------------
    # 1. 加载音频
    # -------------------------
    y, sr = librosa.load(audio_path, sr=sr)

    # -------------------------
    # 2. 提取连续基频 (Hz)
    # -------------------------
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    f0 = np.nan_to_num(f0)  # 无声帧填0

    # -------------------------
    # 3. 转 log 频率
    # -------------------------
    f0_log = np.log(f0 + 1e-6)

    # -------------------------
    # 4. 连续小波变换 (CWT)
    # -------------------------
    if wavelet_widths is None:
        wavelet_widths = np.arange(1, 32)
    cwt_matrix = scipy.signal.cwt(f0_log, scipy.signal.ricker, wavelet_widths)
    cwt_matrix = cwt_matrix.T  # (L, H)

    # -------------------------
    # 5. 量化 (逐尺度)
    # -------------------------
    i_pitch = np.zeros_like(cwt_matrix, dtype=int)
    for h in range(cwt_matrix.shape[1]):
        vals = cwt_matrix[:, h]
        min_val, max_val = np.min(vals), np.max(vals)
        bins = np.linspace(min_val, max_val, n_bins)
        i_pitch[:, h] = np.digitize(vals, bins) - 1

    # -------------------------
    # 6. 帧率重采样
    # -------------------------
    if target_rate is not None:
        orig_rate = 1 / (times[1] - times[0])  # 原始帧率
        L_new = int(round(len(times) * target_rate / orig_rate))
        times_new = np.linspace(times[0], times[-1], L_new)
        i_pitch_resampled = np.zeros((L_new, i_pitch.shape[1]), dtype=int)
        f0_resampled = np.zeros(L_new)
        for h in range(i_pitch.shape[1]):
            interp_func = interp1d(times, i_pitch[:, h], kind='nearest', fill_value="extrapolate")
            i_pitch_resampled[:, h] = interp_func(times_new)
        interp_f0 = interp1d(times, f0, kind='linear', fill_value="extrapolate")
        f0_resampled = interp_f0(times_new)
        times = times_new
        i_pitch = i_pitch_resampled
        f0 = f0_resampled

    # -------------------------
    # 7. 序列长度调整
    # -------------------------
    if seq_len is not None:
        L_curr = i_pitch.shape[0]
        if L_curr < seq_len:
            pad_len = seq_len - L_curr
            i_pitch = np.pad(i_pitch, ((0, pad_len), (0, 0)), mode='constant')
            f0 = np.pad(f0, (0, pad_len), mode='constant')
            # times = np.pad(times, (0, pad_len), mode='linear_ramp', end_values=times[-1]+(np.arange(1,pad_len+1)/target_rate if target_rate else 1))
        else:
            i_pitch = i_pitch[:seq_len, :]
            f0 = f0[:seq_len]
            # times = times[:seq_len]

    # -------------------------
    # 8. 可视化
    # -------------------------
    if show_plot:
                
        plt.figure(figsize=(12,4))
        plt.plot(times, f0, label='f0 (Hz)', color='r')
        plt.title('Pitch Contour')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12,4))
        plt.imshow(i_pitch.T, aspect='auto', origin='lower', extent=[times[0], times[-1], 0, i_pitch.shape[1]], cmap='viridis')
        plt.colorbar(label='Quantized bin index')
        plt.xlabel('Time (s)')
        plt.ylabel('Wavelet scale')
        plt.title('Quantized Pitch Contour (i_pitch)')
        plt.show()

    return i_pitch, f0


def read_and_resample(path, sample_rate):
    waveform, orig_sr = torchaudio.load(path)
    waveform = resample(waveform, orig_sr, sample_rate)
    y = waveform.numpy()
    if len(y.shape) == 2:
        y = y.mean(axis=0)
    return y

def extract_condition(ds, control_type, meta, sample_rate = 44100):
    if control_type == 'loudness':
        seq_len = ds.conditioners_config.loudness.seq_len
        target_rate = ds.conditioners_config.loudness.target_rate
        
        y = read_and_resample(meta['path'], sample_rate)
            
        # 设置hop_length使RMS采样频率为target_rate
        hop_length = int(sample_rate / target_rate)  
        frame_length = 4096  # 可保留原值，也可根据hop_length调整

        # 计算RMS
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # 计算分贝
        min_rms = 1e-6
        rms = np.maximum(rms, min_rms)
        dB = 20 * np.log10(rms / np.max(rms))

        # savgol平滑
        smooth_savgol = savgol_filter(dB, window_length=11, polyorder=3)

        return expand_to(torch.from_numpy(smooth_savgol), len=seq_len)
    
    elif control_type == 'pitch_cwtq':
        # 量化的pitch cwt特征
        seq_len = ds.conditioners_config.pitch_cwtq.seq_len
        target_rate = ds.conditioners_config.pitch_cwtq.target_rate
        
        if 'pitch_cwtq' in meta and meta['pitch_cwtq'].endswith('.npy'):
            i_pitch = np.load(meta['pitch_cwtq'])
            return torch.from_numpy(i_pitch).long()
        
        i_pitch, f0 = extract_pitch_contour(meta['path'], target_rate=target_rate, seq_len=seq_len)
        
        return torch.from_numpy(i_pitch).long()
    
    elif control_type == 'inpaint':
        mode = ds.conditioners_config.inpaint.mode
        sample_rate = {'16k': 16000, '44k': 44100}[mode]
        audio_path = meta['path']

        # 1. 读音频
        waveform, sr = load_audio_mono(audio_path, sample_rate)

        # 2. 获取音频时长（秒）
        duration = waveform.shape[1] / sr

        # 3.1 生成 mask segments
        segs = random_mask_segments(
            duration=duration,
            # min_seg=0.2,
            min_seg=1.0,
            max_seg=5.0,
            min_coverage=0.1,
            max_coverage=0.5,
            min_segments=1,
            # max_segments=5,
            max_segments=1,
            allow_overlap=False,
            seed=None
        )
        # 3.2 将mask部分的waveform置零
        for st, et in segs:
            waveform[:, round(st*sample_rate):round(et*sample_rate)] = 0

        # 4. 返回字典
        return {
            "audio": waveform,  # Tensor [1, T]
            "segments": segs,   # list of (start_sec, end_sec)
        }
        
    elif control_type == 'remove':
        if 'mode' in ds.conditioners_config.remove:
            mode = ds.conditioners_config.remove.mode
        else:
            mode = ds.conditioners_config.reference.mode
        sample_rate = {'16k': 16000, '44k': 44100}[mode]
        audio_path = meta['path']
        waveform, sr = load_audio_mono(audio_path, sample_rate)

        # 1. 随机取背景音
        if "bg_meta" in meta:
            bg_meta = meta["bg_meta"]
        else:
            ds.background_metadata = [d for d in ds.background_metadata if d['duration'] < 10]
            bg_meta = random.sample(ds.background_metadata, 1)[0]
        bg_waveform, _ = load_audio_mono(bg_meta['path'], sample_rate)
        if 'trimmed_duration' in bg_meta:
            st, et = round(bg_meta['start'] * sample_rate), round(bg_meta['end'] * sample_rate)
            bg_waveform = bg_waveform[:, st:et]
        else:
            raise NotImplementedError

        # 2. 随机混合位置
        main_len = waveform.shape[1]
        bg_len = bg_waveform.shape[1]
        if "bg_start" in meta:
            start_sample = round(meta['bg_start'] * sample_rate)
        else:
            if bg_len >= main_len:
                bg_waveform = bg_waveform[:, :main_len]
                start_sample = 0
            else:
                start_sample = random.randint(0, main_len - bg_len)

        end_sample = start_sample + bg_len

        # 3. 对齐能量（RMS 级），确保背景音不低于主音频
        def rms(x):
            return torch.sqrt(torch.mean(x ** 2) + 1e-8)

        rms_wave = rms(waveform)
        rms_bg = rms(bg_waveform)
        scale = (rms_wave / rms_bg).clamp(min=1.0).item()
        bg_gain_factor = meta.get('bg_gain_factor', random.uniform(1.0, 1.3))
        bg_gain = bg_gain_factor * scale
        bg_waveform = bg_waveform * bg_gain

        # 4. 混合
        mixed = waveform.clone()
        mixed[:, start_sample:end_sample] += bg_waveform

        # 5. 峰值归一化（避免削波 + 保持一致）
        peak = max(mixed.abs().max(), waveform.abs().max())
        if peak > 1.0:
            waveform = waveform / peak
            mixed = mixed / peak

        # 6. 时间段（秒）
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate

        # 返回混合结果与时间段
        return {'audio': mixed}, {bg_meta['caption']: [(start_sec, end_sec)]}
    
    elif control_type == 'insert':  # reference, target, events
        if 'mode' in ds.conditioners_config.insert:
            mode = ds.conditioners_config.insert.mode
            secs = ds.conditioners_config.insert.seq_len / ds.conditioners_config.insert.target_rate
        else:
            mode = ds.conditioners_config.reference.mode
        sample_rate = {'16k': 16000, '44k': 44100}[mode]
        audio_path = meta['path']
        waveform, sr = load_audio_mono(audio_path, sample_rate)

        # 0. zero padding
        wav_len = int(sample_rate * secs)
        waveform = waveform[:, :wav_len]
        waveform = torch.nn.functional.pad(waveform, (0, wav_len - waveform.shape[-1]), value=0)

        # 1. 选择背景音
        if "bg_meta" in meta:
            bg_meta = meta["bg_meta"]
        else:
            ds.background_metadata = [d for d in ds.background_metadata if d['duration'] < 10]
            bg_meta = random.sample(ds.background_metadata, 1)[0]
        bg_waveform, _ = load_audio_mono(bg_meta['path'], sample_rate)
        if 'trimmed_duration' in bg_meta:
            st, et = round(bg_meta['start'] * sample_rate), round(bg_meta['end'] * sample_rate)
            bg_waveform = bg_waveform[:, st:et]
            # print_once("Use trimmed", bg_meta['start'], bg_meta['end'])
        else:
            raise NotImplementedError

        # 2. 随机插入位置
        main_len = waveform.shape[1]
        bg_len = bg_waveform.shape[1]
        if "bg_start" in meta:
            start_sample = round(meta['bg_start'] * sample_rate)
        else:
            if bg_len >= main_len:
                bg_waveform = bg_waveform[:, :main_len]
                start_sample = 0
            else:
                start_sample = random.randint(0, main_len - bg_len)

        end_sample = start_sample + bg_len

        # 3. 对齐能量（RMS级），确保背景音至少不低于主音频
        def rms(x):
            return torch.sqrt(torch.mean(x ** 2) + 1e-8)

        rms_wave = rms(waveform)
        rms_bg = rms(bg_waveform)
        scale = (rms_wave / rms_bg).clamp(min=1.0).item()
        bg_gain_factor = meta.get('bg_gain_factor', random.uniform(1.0, 1.3))
        bg_gain = bg_gain_factor * scale
        bg_waveform = bg_waveform * bg_gain

        # 4. 混合
        mixed = waveform.clone()
        mixed[:, start_sample:end_sample] += bg_waveform

        # 5. 峰值归一化（避免削波，同时保持waveform与mixed可比）
        peak = max(mixed.abs().max(), waveform.abs().max())
        if peak > 1.0:
            waveform = waveform / peak
            mixed = mixed / peak

        # 6. 计算时间（秒）
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate

        # 返回 (reference, target, event)
        return {'audio': waveform}, {'audio': mixed}, {bg_meta['caption']: [(start_sec, end_sec)]}

    elif control_type == 'f0':
        waveform, sr = torchaudio.load(path)
        y = waveform.numpy()
        if len(y.shape) == 2:
            y = y.mean(axis=0)
        sr = ds.sample_rate
        duration = len(y) / sr

        # 设置 hop_length，使 f0 长度为 21.5 * duration
        hop_length = int(sr / 21.5)

        # 使用 YIN 算法提取基频
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr, hop_length=hop_length)

        # 高斯平滑处理
        sigma = 1
        smoothed_f0 = gaussian_filter1d(f0, sigma=sigma)

        return torch.from_numpy(smoothed_f0).float()

    elif control_type == 'centroid':
        waveform, sr = torchaudio.load(path)
        y = waveform.numpy()
        if len(y.shape) == 2:
            y = y.mean(axis=0)
        sr = ds.sample_rate
        target_rate = 21.5
        hop_length = int(sr / target_rate)

        # 提取谱质心（shape: [1, T]）
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid = np.log(centroid+1e-6)

        # 时间轴
        frames = np.arange(len(centroid))
        times = frames * hop_length / sr

        # # 高斯滤波
        # centroid_gauss = gaussian_filter1d(centroid, sigma=5)
        # 中值滤波（kernel size 5，必须为奇数）
        centroid_median = medfilt(centroid, kernel_size=5)

        return torch.from_numpy(centroid_median).float()

    elif control_type == 'vocal':
        path = meta["vocal"]

        waveform, sr = torchaudio.load(path)
        waveform = waveform[:2, ...]
        if waveform.shape[0] == 1:
            waveform = torch.cat([waveform, waveform], dim=0)

        # 重采样
        if sr != ds.sample_rate:
            waveform = resample(waveform, orig_freq=sr, new_freq=ds.sample_rate)
            sr = ds.sample_rate

        total_duration = waveform.shape[-1] / sr

        if ds.chunk_dur is not None:
            chunk_size = int(ds.chunk_dur * sr)
            current_size = waveform.shape[-1]

            if current_size > chunk_size:
                # 裁剪
                max_offset = current_size - chunk_size
                offset = meta['offset']
                waveform = waveform[:, offset:offset + chunk_size]
                start_sec = offset / sr
            elif current_size < chunk_size:
                # 右侧 zero-padding
                pad_size = chunk_size - current_size
                pad = torch.zeros((waveform.shape[0], pad_size), dtype=waveform.dtype)
                waveform = torch.cat([waveform, pad], dim=-1)
                start_sec = 0.0
            else:
                # 刚好等于 chunk_size，无需处理
                start_sec = 0.0
        else:
            start_sec = 0.0
        return waveform
    elif control_type == 'events':
        return meta['events']
    elif control_type == 'chord':
        if isinstance(meta['chord'], str):
            with open(meta['chord']) as f:
                lines = f.readlines()
            chord_data = []
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                st, et, ch = line.strip().split()
                chord_data.append((float(st), float(et), ch))
            return chord_data
        else:
            return meta['chord']
    else:
        raise NotImplementedError
    
def extract_all_conditions(ds, meta, sample_rate = 44100):
    ret = {}
    for ct in ds.control_types:
        if ct == 'reference':
            continue # reference 将在remove中完成初始化
        elif ct in ['remove']:
            reference, ret[ct] = extract_condition(ds, ct, meta, sample_rate=sample_rate)
            if 'reference' in ds.control_types: 
                # 如果有reference controlnet，则直接将reference给到一个条件
                assert ret.get('reference', None) is None
                ret['reference'] = reference
            else:
                # 否则，直接把reference给到输入中
                ret[ct] = {"events": ret[ct], "reference": reference}
        elif ct in ['insert']:
            reference, target, events = extract_condition(ds, ct, meta, sample_rate=sample_rate)
            assert 'reference' not in ds.control_types
            ret[ct] = {"events": events, "reference": reference, "target": target}
        elif ct == 'insert_events':
            ret[ct] = ret['insert']['events']
        else:
            ret[ct] = extract_condition(ds, ct, meta, sample_rate=sample_rate)
    return ret