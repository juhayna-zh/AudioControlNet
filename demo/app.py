import gradio as gr
import torch

# -----------------------------
# Placeholder T2A generation
# -----------------------------

def generate_audio(text, duration, guidance_scale, seed):
    """
    Placeholder function for Text-to-Audio generation.
    Replace this with your actual Audio ControlNet inference code.
    """
    if seed != -1:
        torch.manual_seed(seed)

    # TODO: replace with real waveform generation
    # Here we just generate silence as a placeholder
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    waveform = torch.zeros(num_samples)

    return (sample_rate, waveform.numpy())


# -----------------------------
# Gradio Interface
# -----------------------------

blue_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
)

with gr.Blocks(theme=blue_theme, title="Audio ControlNet – Text to Audio") as demo:
    gr.Markdown(
        """
        # 🎵 Audio ControlNet
        ## Text-to-Audio Generation (T2A)
        Generate audio from text prompts using **Audio ControlNet**.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="A calm ambient soundscape with soft pads and distant piano",
                lines=4,
            )

            duration = gr.Slider(
                minimum=1.0,
                maximum=30.0,
                value=10.0,
                step=0.5,
                label="Duration (seconds)",
            )

            guidance_scale = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=5.0,
                step=0.1,
                label="Guidance Scale",
            )

            seed = gr.Number(
                value=-1,
                precision=0,
                label="Random Seed (-1 for random)",
            )

            generate_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
            )

    generate_btn.click(
        fn=generate_audio,
        inputs=[text_prompt, duration, guidance_scale, seed],
        outputs=audio_output,
    )

    gr.Markdown(
        """
        ---
        **Note**: This is a basic T2A interface. Control signals (e.g. rhythm, loudness, pitch)
        can be added later as additional inputs for Audio ControlNet.
        """
    )


if __name__ == "__main__":
    demo.launch()
