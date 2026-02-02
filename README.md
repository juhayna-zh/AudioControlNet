# Audio ControlNet 

This is the official repository for the paper "Audio ControlNet for Fine-Grained Audio Generation and Editing".

Audio ControlNet enables fine-grained control over audio generation through multiple conditioning mechanisms including loudness, pitch, and sound events. This allows for precise audio synthesis and editing with controllable attributes.

## 🎯 Supported Control Types

- **Loudness**: Controls energy dynamics and volume variations
- **Pitch**: Controls fundamental frequency contours
- **Events**: Controls timing and occurrence of specific sound categories


## 🚀 Quick Start

### Installation

First, install the package in development mode:

```bash
pip install -e .
```

This will install all required dependencies including PyTorch, librosa, transformers, and other necessary packages.

### Basic Usage

```python
from audio_controlnet.infer import AudioControlNet

# Generate audio with text prompt
caption = "A man is speaking while walking, music is playing, and sound effects are heard."
```

## 📖 Usage Examples

### 1. Loudness Control

Control the energy dynamics of generated audio using a reference audio file:

```python
model = AudioControlNet.from_pretrained('juhayna/T2A-Adapter-loudness-v1.0')

res = model.infer(
    caption=caption,
    control={'loudness': model.prepare_loudness('./reference.flac')}
)
torchaudio.save('./output/loudness_controlled.wav', res.audio, res.sample_rate)
```

### 2. Pitch Control

Control the pitch contour of generated audio:

```python
model = AudioControlNet.from_pretrained('juhayna/T2A-Adapter-pitch-v1.0')

res = model.infer(
    caption=caption,
    control={'pitch': model.prepare_pitch('./reference.flac')}
)
torchaudio.save('./output/pitch_controlled.wav', res.audio, res.sample_rate)
```

### 3. Sound Events Control

Control the timing and occurrence of specific sound events:

```python
model = AudioControlNet.from_pretrained('juhayna/T2A-Adapter-events-v1.0')

# Define sound events with timestamps
events = {
    "Sound effect": [[0.5, 1.6], [4.1, 5.4], [7.0, 8.5]], 
    "Male speech, man speaking": [[5.0, 10.0]], 
    "Music": [[9.0, 10.0]]
}

# Generate audio with events conditioning
res = model.infer(
    caption=caption,
    control={'events': events}
)
torchaudio.save('./output/events_controlled.wav', res.audio, res.sample_rate)
```

### 4. Multi-Control Generation

Combine multiple control conditions for fine-grained control:

```python
model = AudioControlNet.from_multi_controlnets([
    'juhayna/T2A-Adapter-loudness-v1.0',
    'juhayna/T2A-Adapter-pitch-v1.0',
    'juhayna/T2A-Adapter-events-v1.0',
])

# Use multiple controls simultaneously
res = model.infer(
    caption=caption,
    control={
        'loudness': model.prepare_loudness('./reference.flac'),
        'pitch': model.prepare_pitch('./reference.flac'),
        'events': events
    }
)
torchaudio.save('./output/multi_controlled.wav', res.audio, res.sample_rate)
```


## 🎛️ Web Interface

Launch the Gradio web interface for interactive audio generation:

```bash
gradio app.py
```

The web interface provides:
- Text-to-audio generation
- Interactive control condition setup
- Real-time visualization of loudness, pitch, and events
- Audio playback and download

## 🚂 Training

Coming Soon.

## 📋 Citation

Coming soon.

## 🙏 Acknowledgements

We specially thank the following repositories:

- [MeanAudio](https://github.com/xiquan-li/MeanAudio)
- [av-benchmark](https://github.com/hkchengrex/av-benchmark)
- [AudioComposer](https://github.com/lavendery/AudioComposer)