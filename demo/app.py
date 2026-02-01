import gradio as gr
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json5
import torchaudio
import tempfile
import os
from audio_controlnet.infer import AudioControlNet

MAX_DURATION = 10.0  # seconds

# -----------------------------
# Feature extraction utilities
# -----------------------------
def process_audio_clip(audio):
    if audio is None:
        return None
    sr, y = audio
    y = y.astype(np.float32)
    num_samples = int(MAX_DURATION * sr)
    if y.shape[0] > num_samples:
        y = y[:num_samples]
    elif y.shape[0] < num_samples:
        padding = num_samples - y.shape[0]
        y = np.pad(y, (0, padding))
    return (sr, y)

def extract_loudness(audio):
    audio = process_audio_clip(audio)
    if audio is None:
        return None
    sr, y = audio
    if y.ndim == 2:
        y = y.mean(axis=1)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, rms)
    ax.set_title("Loudness (RMS)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    fig.tight_layout()
    return fig

def extract_pitch(audio):
    audio = process_audio_clip(audio)
    if audio is None:
        return None
    sr, y = audio
    if y.ndim == 2:
        y = y.mean(axis=1)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
    )
    times = librosa.times_like(f0, sr=sr)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, f0)
    ax.set_title("Pitch (F0 contour)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return fig

def visualize_events(json_str):
    try:
        events = json5.loads(json_str)
    except:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    cmap = cm.get_cmap("tab10")
    labels = list(events.keys())
    color_map = {label: cmap(i % 10) for i, label in enumerate(labels)}

    for i, (label, intervals) in enumerate(events.items()):
        color = color_map[label]
        for start, end in intervals:
            if start >= MAX_DURATION:
                continue
            end = min(end, MAX_DURATION)
            ax.barh(i, end - start, left=start, height=0.5, color=color)

    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Sound Events Timeline")
    ax.set_xlim(0, MAX_DURATION)
    fig.tight_layout()
    return fig

# -----------------------------
# AudioControlNet Initialization
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = AudioControlNet.from_multi_controlnets(
    [
        "juhayna/T2A-Adapter-loudness-v1.0",
        "juhayna/T2A-Adapter-pitch-v1.0",
        "juhayna/T2A-Adapter-events-v1.0",
    ],
    device=DEVICE,
)

# -----------------------------
# Temporary WAV utility
# -----------------------------
def save_temp_wav(audio):
    if audio is None:
        return None
    sr, y = audio
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = torch.from_numpy(y).float().unsqueeze(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, y, sr)
    return tmp.name

# -----------------------------
# Generate audio
# -----------------------------
def generate_audio(text, cond_loudness, cond_pitch, cond_events):
    control = {}
    temp_files = []

    try:
        if cond_loudness is not None:
            wav_path = save_temp_wav(cond_loudness)
            temp_files.append(wav_path)
            control["loudness"] = model.prepare_loudness(wav_path)

        elif cond_pitch is not None:
            wav_path = save_temp_wav(cond_pitch)
            temp_files.append(wav_path)
            control["pitch"] = model.prepare_pitch(wav_path)

        elif cond_events:
            events = json5.loads(cond_events)
            control["events"] = events

        with torch.no_grad():
            res = model.infer(
                caption=text,
                control=control if len(control) > 0 else None,
            )

        audio = res.audio.squeeze(0).cpu().numpy()
        sr = res.sample_rate
        return (sr, audio)

    finally:
        for f in temp_files:
            if f and os.path.exists(f):
                os.remove(f)

# -----------------------------
# Gradio Interface
# -----------------------------
blue_theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate")

EVENTS_PLACEHOLDER = '''
// example
{
    "Video game sound": [[0.0, 10.0]], 
    "Male speech, man speaking": [[0.015, 3.829], [4.293, 4.875], [5.089, 7.349], [8.071, 9.978]]
}
'''.strip()

with gr.Blocks(theme=blue_theme, title="Audio ControlNet – Text to Audio") as demo:
    gr.Markdown("""
        # 🎵 Audio ControlNet
        ## Text-to-Audio Generation with Conditions
        Base T2A interface with conditional inputs for **Audio ControlNet**.
    """)
    gr.HTML("""
    <style>
    .plot-small { height: 250px !important; }
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="A calm ambient soundscape with soft pads and distant piano",
                lines=4,
            )

            with gr.Tabs() as tabs:
                with gr.Tab("Loudness") as tab_loudness:
                    with gr.Row():
                        with gr.Column(scale=1):
                            loudness_audio = gr.Audio(label="Loudness Reference Audio (up to 10 sec)", type="numpy")
                        with gr.Column(scale=1):
                            loudness_plot = gr.Plot(label="Loudness Curve (Reference Audio)", elem_classes="plot-small")

                with gr.Tab("Pitch") as tab_pitch:
                    with gr.Row():
                        with gr.Column(scale=1):
                            pitch_audio = gr.Audio(label="Pitch Reference Audio (up to 10 sec)", type="numpy")
                        with gr.Column(scale=1):
                            pitch_plot = gr.Plot(label="Pitch Curve (Reference Audio)", elem_classes="plot-small")

                with gr.Tab("Sound Events") as tab_events:
                    with gr.Row():
                        with gr.Column(scale=1):
                            sound_events = gr.Textbox(label="Sound Events (JSON)", placeholder=EVENTS_PLACEHOLDER, lines=8)
                        with gr.Column(scale=1):
                            events_plot = gr.Plot(label="Sound Events Roll", elem_classes="plot-small")

            generate_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio", type="numpy")

    loudness_audio.change(fn=extract_loudness, inputs=loudness_audio, outputs=loudness_plot)
    pitch_audio.change(fn=extract_pitch, inputs=pitch_audio, outputs=pitch_plot)
    sound_events.change(fn=visualize_events, inputs=sound_events, outputs=events_plot)

    generate_btn.click(
        fn=generate_audio,
        inputs=[text_prompt, loudness_audio, pitch_audio, sound_events],
        outputs=audio_output
    )

    tab_loudness.select(lambda: (None, None), [], [pitch_audio, sound_events])
    tab_pitch.select(lambda: (None, None), [], [loudness_audio, sound_events])
    tab_events.select(lambda: (None, None), [], [loudness_audio, pitch_audio])

    gr.Markdown("""
        ---
        **Control Inputs**
        - **Loudness**: reference audio controlling energy / dynamics
        - **Pitch**: reference audio controlling pitch contour
        - **Sound Events**: symbolic event-level constraints in JSON format
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
