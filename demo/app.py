import gradio as gr
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json5

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
    
    # 生成颜色映射，保证同一事件颜色一致
    cmap = cm.get_cmap("tab10")  # 20种颜色循环使用
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
# Placeholder T2A generation
# -----------------------------
def generate_audio(text, cond_loudness, cond_pitch, cond_events):
    sample_rate = 16000
    duration = int(MAX_DURATION)
    waveform = torch.zeros(sample_rate * duration)
    return (sample_rate, waveform.numpy())

# -----------------------------
# Gradio Interface
# -----------------------------
blue_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
)

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
                # -----------------------------
                # Loudness Tab
                # -----------------------------
                with gr.Tab("Loudness") as tab_loudness:
                    with gr.Row():
                        with gr.Column(scale=1):
                            loudness_audio = gr.Audio(label="Loudness Reference Audio", type="numpy")
                        with gr.Column(scale=1):
                            loudness_plot = gr.Plot(label="Loudness Curve (Reference Audio)", elem_classes="plot-small")

                # -----------------------------
                # Pitch Tab
                # -----------------------------
                with gr.Tab("Pitch") as tab_pitch:
                    with gr.Row():
                        with gr.Column(scale=1):
                            pitch_audio = gr.Audio(label="Pitch Reference Audio", type="numpy")
                        with gr.Column(scale=1):
                            pitch_plot = gr.Plot(label="Pitch Curve (Reference Audio)", elem_classes="plot-small")

                # -----------------------------
                # Sound Events Tab
                # -----------------------------
                with gr.Tab("Sound Events") as tab_events:
                    with gr.Row():
                        with gr.Column(scale=1):
                            sound_events = gr.Textbox(label="Sound Events (JSON)", placeholder=EVENTS_PLACEHOLDER, lines=8)
                        with gr.Column(scale=1):
                            events_plot = gr.Plot(label="Sound Events Roll", elem_classes="plot-small")

            generate_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio", type="numpy")

    # -----------------------------
    # 上传音频 / JSON 绘制曲线
    # -----------------------------
    loudness_audio.change(fn=extract_loudness, inputs=loudness_audio, outputs=loudness_plot)
    pitch_audio.change(fn=extract_pitch, inputs=pitch_audio, outputs=pitch_plot)
    sound_events.change(fn=visualize_events, inputs=sound_events, outputs=events_plot)

    # -----------------------------
    # 生成按钮
    # -----------------------------
    generate_btn.click(fn=generate_audio,
                       inputs=[text_prompt, loudness_audio, pitch_audio, sound_events],
                       outputs=audio_output)

    # -----------------------------
    # Tab 切换清空其他条件
    # -----------------------------
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
    demo.launch()
