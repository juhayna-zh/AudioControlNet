import gradio as gr
try:
    import spaces
    require_gpu = spaces.GPU
except:
    require_gpu = lambda f: f
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json5
import torchaudio
import tempfile
import os
import random
from audio_controlnet.infer import AudioControlNet

import logging
logging.getLogger("gradio").setLevel(logging.WARNING)

MAX_DURATION = 10.0  # seconds

# -----------------------------
# Random Examples Data
# -----------------------------
RANDOM_EXAMPLES = [
  {
    "caption": "People speak and clap, a child speaks and a camera clicks.",
    "events": {
      "Female speech, woman speaking": [[0.0, 3.969], [7.913, 8.157], [8.189, 9.654]],
      "Child speech, kid speaking": [[9.724, 10.0]]
    }
  },
  {
    "caption": "Background noise, tapping, and cat sounds are interspersed with purring.",
    "events": {
      "Cat": [[0.978, 2.291], [9.032, 10.0]]
    }
  },
  {
    "caption": "Water flows and dishes clatter with child speech and laughter.",
    "events": {
      "Child speech, kid speaking": [[0.0, 1.503], [1.732, 2.12], [2.942, 3.541], [7.803, 8.493]],
      "Dishes, pots, and pans": [[1.983, 2.156], [3.175, 3.298], [4.774, 5.076], [5.711, 5.834], [6.076, 6.24], [6.423, 7.012]],
      "Male speech, man speaking": [[8.547, 9.557]],
      "Water tap, faucet": [[0.0, 10.0]]
    }
  },
  {
    "caption": "Speech babble and clattering dishes and silverware can be heard, along with a child's voice.",
    "events": {
      "Dishes, pots, and pans": [[0.85, 0.969], [1.386, 1.504], [7.717, 7.874]],
      "Male speech, man speaking": [[0.748, 1.173]],
      "Cutlery, silverware": [[4.693, 4.843], [5.299, 5.52]],
      "Female speech, woman speaking": [[1.63, 3.409]],
      "Child speech, kid speaking": [[8.756, 9.354]]
    }
  },
  {
    "caption": "A man is speaking, with background sounds of wind and a river, and another man sighing and speaking.", 
    "events": {"Male speech, man speaking": [[0.0, 7.851], [8.903, 9.129], [9.328, 9.98]], "Conversation": [[0.0, 9.98]], "Wind": [[0.0, 9.98]], "Stream, river": [[0.0, 9.98]], "Sigh": [[8.157, 8.707]]}
  },
  {
    "caption": "Wind noise and cowbell are heard twice.", 
    "events": {"Wind noise (microphone)": [[0.0, 1.15], [2.378, 2.961]], "Cowbell": [[0.0, 10.0]]}
  },
  {
    "caption": "There are mechanisms, bird calls, clicking, and male speech.", 
    "events": {"Mechanisms": [[0.0, 10.0]], "Bird vocalization, bird call, bird song": [[1.122, 1.423]], "Clicking": [[1.139, 1.238], [4.737, 4.858]], "Male speech, man speaking": [[1.95, 2.875], [5.182, 5.795], [6.113, 6.807], [7.386, 8.138], [8.236, 8.803], [9.427, 10.0]]}
  },
  {
    "caption": "Propeller noise and a sound effect.", 
    "events": {"Propeller, airscrew": [[1.779, 10.0]], "Sound effect": [[1.811, 2.868]]}
  },
  {
    "caption": "Women converse and laugh in a noisy crowd.", 
    "events": {"Female speech, woman speaking": [[0.0, 1.669], [2.097, 2.976], [4.66, 8.98]], "Conversation": [[0.0, 9.379]], "Background noise": [[0.0, 9.379]], "Generic impact sounds": [[0.096, 0.318], [3.707, 3.944], [6.107, 6.314], [7.584, 7.695], [8.256, 8.367]], "Laughter": [[1.573, 2.947], [4.461, 6.174], [9.002, 9.364]], "Crowd": [[1.573, 2.954], [4.512, 6.129], [9.002, 9.379]], "Tick": [[1.691, 1.795], [4.276, 4.372]], "Sound effect": [[3.212, 4.416]]}
  }
]
def build_events_json_text(events):
    ret = ''
    for key,times in events.items():
        ret += f'    "{key}": {times},\n'
    ret = ret.strip(',')
    return '{\n'+ret+'}'

def generate_random_example():
    """Generate a random example with caption and sound events"""
    example = random.choice(RANDOM_EXAMPLES)
    events_json = build_events_json_text(example["events"])
    return example["caption"], events_json

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
@require_gpu
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

# Generate initial random example for page load
initial_caption, initial_events = generate_random_example()

CAPTION_PLACEHOLDER = 'Water flows and dishes clatter with child speech and laughter.'

EVENTS_PLACEHOLDER = '''
// example
{
    "Child speech, kid speaking": [[0.0, 1.503], [1.732, 2.12], [2.942, 3.541], [7.803, 8.493]],
    "Dishes, pots, and pans": [[1.983, 2.156], [3.175, 3.298], [4.774, 5.076], [5.711, 5.834], [6.076, 6.24], [6.423, 7.012]],
    "Water tap, faucet": [[0.0, 10.0]]
}
'''.strip()

with gr.Blocks(theme=blue_theme, title="Audio ControlNet – Text to Audio") as demo:
    gr.Markdown("""
        # 🎵 Audio ControlNet
        ## Fine-Grained Text-to-Audio Generation with Conditions
        T2A GUI interface with conditional inputs for **Audio ControlNet**.
    """)
    gr.HTML("""
    <style>
    .plot-small { height: 280px !important; }
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_prompt = gr.Textbox(
                label="Text Prompt",
                placeholder=CAPTION_PLACEHOLDER,
                lines=4,
                value=initial_caption,
            )

            with gr.Tabs() as tabs:
                with gr.Tab("Sound Events") as tab_events:
                    with gr.Row():
                        with gr.Column(scale=1):
                            sound_events = gr.Textbox(label="Sound Events (JSON)", placeholder=EVENTS_PLACEHOLDER, lines=8, value=initial_events)
                            random_example_btn = gr.Button("🎲 Random Example", variant="primary", size="sm")
                        with gr.Column(scale=1):
                            events_plot = gr.Plot(label="Sound Events Roll", elem_classes="plot-small")
                            
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

            generate_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio", type="numpy")

    loudness_audio.change(fn=extract_loudness, inputs=loudness_audio, outputs=loudness_plot)
    pitch_audio.change(fn=extract_pitch, inputs=pitch_audio, outputs=pitch_plot)
    sound_events.change(fn=visualize_events, inputs=sound_events, outputs=events_plot)
    
    # Initialize events plot with the initial random example
    demo.load(fn=lambda: visualize_events(initial_events), inputs=[], outputs=events_plot)
    
    # Random example button event
    random_example_btn.click(
        fn=generate_random_example,
        inputs=[],
        outputs=[text_prompt, sound_events]
    )

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
    demo.launch(server_name="0.0.0.0", quiet=True)
