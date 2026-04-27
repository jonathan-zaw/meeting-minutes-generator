
import os
import torch
import gradio as gr
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# Auth — reads token from environment variable
hf_token = os.environ.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Config
WHISPER_MODEL = "openai/whisper-medium.en"
LLAMA_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"

# Lazy loaders
_whisper_pipe = None
_llama_model  = None
_llama_tok    = None

def _load_whisper():
    global _whisper_pipe
    if _whisper_pipe is None:
        _whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            dtype=torch.float16,
            device="cuda",
            return_timestamps=True,
        )
    return _whisper_pipe

def _load_llama():
    global _llama_model, _llama_tok
    if _llama_model is None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        _llama_tok = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        _llama_tok.pad_token = _llama_tok.eos_token
        _llama_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL,
            device_map="auto",
            quantization_config=quant_config,
        )
    return _llama_model, _llama_tok

def transcribe_audio(audio_path):
    pipe = _load_whisper()
    result = pipe(audio_path)
    return result["text"]

def generate_minutes(transcription, max_new_tokens=2000):
    model, tokenizer = _load_llama()

    system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""
    user_prompt = f"""
Below is an extract transcript of a meeting.
Please write minutes in markdown without code blocks, including:
1. **Summary**: Include the meeting date, location, and a list of attendees.
2. **Discussion Points**: Summarize the core topics debated or presented.
3. **Key Takeaways**: Highlight the most significant conclusions or consensus reached.
4. **Action Items**: Create a table or list of tasks, including the specific owner/official responsible for each.

Transcription:
{transcription}
"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)

    generated_ids = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def run_pipeline(audio_file, max_tokens, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")
    progress(0.1, desc="Transcribing audio with Whisper …")
    transcription = transcribe_audio(audio_file)
    progress(0.5, desc="Generating meeting minutes with Llama …")
    minutes = generate_minutes(transcription, max_new_tokens=int(max_tokens))
    progress(1.0, desc="Done!")
    return transcription, minutes

def transcribe_only(audio_file, progress=gr.Progress()):
    if audio_file is None:
        raise gr.Error("Please upload an audio file first.")
    progress(0.2, desc="Transcribing …")
    text = transcribe_audio(audio_file)
    progress(1.0, desc="Done!")
    return text

with gr.Blocks(title="Meeting Minutes Generator") as demo:
    gr.Markdown("""
    #  Meeting Minutes Generator
    Upload your meeting audio. The app will:
    1. **Transcribe** it with *Whisper medium (en)*
    2. **Generate structured minutes** with *Llama-3.2-3B-Instruct (4-bit)*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Meeting Audio",
                type="filepath",
                sources=["upload"],
            )
            max_tokens_slider = gr.Slider(
                minimum=512, maximum=4096, value=2000,
                step=128, label="Max output tokens",
            )
            with gr.Row():
                transcribe_btn = gr.Button("Transcribe Only", variant="secondary")
                run_btn        = gr.Button("Transcribe + Generate Minutes", variant="primary")

        with gr.Column(scale=2):
            transcription_box = gr.Textbox(
                label="Transcription", lines=10,
                placeholder="Transcription will appear here …",
                show_copy_button=True,
            )
            minutes_box = gr.Markdown(
                label="Meeting Minutes",
                value="*Meeting minutes will appear here …*",
            )

    transcribe_btn.click(fn=transcribe_only, inputs=[audio_input], outputs=[transcription_box])
    run_btn.click(fn=run_pipeline, inputs=[audio_input, max_tokens_slider], outputs=[transcription_box, minutes_box])

demo.launch(share=True)
