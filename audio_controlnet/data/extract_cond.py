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
    n = tensor.size(0)
    
    if n >= len:
        return tensor[:len]
    else:
        pad_times = len - n
        paded_part = torch.ones((pad_times,)) * pad_elem
        expanded_tensor = torch.cat([tensor, paded_part])
        return expanded_tensor
    
def load_audio_mono(audio_path, sample_rate=16000):
    waveform, sr = torchaudio.load(audio_path)  # waveform: [channels, T]
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
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
    target_rate: float = None,  # Hz
    seq_len: int = None,
    show_plot: bool = False, 
):
    """
    Extract the pitch contour as described in the paper, with support for frame rate resampling
    and sequence length adjustment.

    Args:
        audio_path (str): Path to the audio file.
        sr (int): Sampling rate.
        fmin (float): Minimum frequency for pitch search.
        fmax (float): Maximum frequency for pitch search.
        n_bins (int, optional): Number of quantization bins (default 256).
        wavelet_widths (list, optional): Range of wavelet scales (default [1..32]).
        target_rate (float, optional): Target frame rate in Hz; None means no resampling.
        seq_len (int, optional): Target sequence length.
        show_plot (bool, optional): Whether to display a heatmap visualization.

    Returns:
        i_pitch (ndarray): Quantized pitch sequence of shape (L × H).
        f0 (ndarray): Original pitch trajectory in Hz, length L.
        times (ndarray): Time axis in seconds, length L.
    """

    # -------------------------
    # 1. Load Audio
    # -------------------------
    y, sr = librosa.load(audio_path, sr=sr)

    # -------------------------
    # 2. Extract F0 (Hz)
    # -------------------------
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr
    )
    times = librosa.times_like(f0, sr=sr)
    f0 = np.nan_to_num(f0) 

    # -------------------------
    # 3. Convert to log F0
    # -------------------------
    f0_log = np.log(f0 + 1e-6)

    # -------------------------
    # 4. CWT
    # -------------------------
    if wavelet_widths is None:
        wavelet_widths = np.arange(1, 32)
    cwt_matrix = scipy.signal.cwt(f0_log, scipy.signal.ricker, wavelet_widths)
    cwt_matrix = cwt_matrix.T  # (L, H)

    # -------------------------
    # 5. Quantize
    # -------------------------
    i_pitch = np.zeros_like(cwt_matrix, dtype=int)
    for h in range(cwt_matrix.shape[1]):
        vals = cwt_matrix[:, h]
        min_val, max_val = np.min(vals), np.max(vals)
        bins = np.linspace(min_val, max_val, n_bins)
        i_pitch[:, h] = np.digitize(vals, bins) - 1

    # -------------------------
    # 6. Resample
    # -------------------------
    if target_rate is not None:
        orig_rate = 1 / (times[1] - times[0])  
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
    # 7. Adjust seq length
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
    # 8. Visualize
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
            
        # Set hop_length to make the RMS sampling frequency equal to target_rate
        hop_length = int(sample_rate / target_rate)  
        frame_length = 4096 

        # calc RMS
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # calc DB
        min_rms = 1e-6
        rms = np.maximum(rms, min_rms)
        dB = 20 * np.log10(rms / np.max(rms))

        # savgol filter
        smooth_savgol = savgol_filter(dB, window_length=11, polyorder=3)

        return expand_to(torch.from_numpy(smooth_savgol), len=seq_len)
    
    elif control_type == 'pitch':
        # quantized pitch cwt
        seq_len = ds.conditioners_config.pitch.seq_len
        target_rate = ds.conditioners_config.pitch.target_rate
        
        if 'pitch' in meta and meta['pitch'].endswith('.npy'):
            i_pitch = np.load(meta['pitch'])
            return torch.from_numpy(i_pitch).long()
        
        i_pitch, f0 = extract_pitch_contour(meta['path'], target_rate=target_rate, seq_len=seq_len)
        
        return torch.from_numpy(i_pitch).long()
    
    elif control_type == 'inpaint':
        mode = ds.conditioners_config.inpaint.mode
        sample_rate = {'16k': 16000, '44k': 44100}[mode]
        audio_path = meta['path']

        # 1. read audio
        waveform, sr = load_audio_mono(audio_path, sample_rate)

        # 2. Get audio duration (seconds)
        duration = waveform.shape[1] / sr

        # 3.1 generate mask segments
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
        # 3.2 set masked part to 0
        for st, et in segs:
            waveform[:, round(st*sample_rate):round(et*sample_rate)] = 0

        # 4. Return
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

        # 1. Select background
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

        # 2.  random insert
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

        # 3. Align the energy (RMS level) to ensure that the background is not lower than the main audio
        def rms(x):
            return torch.sqrt(torch.mean(x ** 2) + 1e-8)

        rms_wave = rms(waveform)
        rms_bg = rms(bg_waveform)
        scale = (rms_wave / rms_bg).clamp(min=1.0).item()
        bg_gain_factor = meta.get('bg_gain_factor', random.uniform(1.0, 1.3))
        bg_gain = bg_gain_factor * scale
        bg_waveform = bg_waveform * bg_gain

        # 4. Mix
        mixed = waveform.clone()
        mixed[:, start_sample:end_sample] += bg_waveform

        # 5. Normalization
        peak = max(mixed.abs().max(), waveform.abs().max())
        if peak > 1.0:
            waveform = waveform / peak
            mixed = mixed / peak

        # 6. Time
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate

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

        # 1. Select background
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

        # 2. random insert
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

        # 3. Align the energy (RMS level) to ensure that the background is not lower than the main audio
        def rms(x):
            return torch.sqrt(torch.mean(x ** 2) + 1e-8)

        rms_wave = rms(waveform)
        rms_bg = rms(bg_waveform)
        scale = (rms_wave / rms_bg).clamp(min=1.0).item()
        bg_gain_factor = meta.get('bg_gain_factor', random.uniform(1.0, 1.3))
        bg_gain = bg_gain_factor * scale
        bg_waveform = bg_waveform * bg_gain

        # 4. Mix
        mixed = waveform.clone()
        mixed[:, start_sample:end_sample] += bg_waveform

        # 5. Normalization
        peak = max(mixed.abs().max(), waveform.abs().max())
        if peak > 1.0:
            waveform = waveform / peak
            mixed = mixed / peak

        # 6. Time
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate

        return {'audio': waveform}, {'audio': mixed}, {bg_meta['caption']: [(start_sec, end_sec)]}
    elif control_type == 'events':
        return meta['events']
    else:
        raise NotImplementedError
    
def extract_all_conditions(ds, meta, sample_rate = 44100):
    ret = {}
    for ct in ds.control_types:
        if ct == 'reference':
            continue # reference will be initialized later
        elif ct in ['remove']:
            reference, ret[ct] = extract_condition(ds, ct, meta, sample_rate=sample_rate)
            if 'reference' in ds.control_types: 
                # If there is a reference controlnet, directly assign the reference to a condition
                assert ret.get('reference', None) is None
                ret['reference'] = reference
            else:
                # Otherwise, directly provide the reference to the input
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