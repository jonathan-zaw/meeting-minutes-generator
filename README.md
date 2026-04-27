# Meeting Minutes Generator

An AI-powered application that transcribes meeting audio and automatically generates structured meeting minutes using open-source models.

---

## Overview

This project takes an audio recording of a meeting and produces a clean, structured summary including discussion points, key takeaways, and action items. It runs entirely on open-source models without requiring any paid API.

#### How it works

1. Audio is transcribed using **OpenAI Whisper**
2. The transcription is passed to **Llama 3.2** which generates formatted meeting minutes

---

## Models Used

| Model | Purpose |
|---|---|
| `openai/whisper-medium.en` | Speech-to-text transcription |
| `meta-llama/Llama-3.2-3B-Instruct` | Meeting minutes generation |

Llama is loaded in **4-bit quantization** using BitsAndBytes to reduce GPU memory usage while maintaining output quality.

---

## Features

- Upload any meeting audio file and get a full transcription
- Automatically generates minutes with Summary, Discussion Points, Key Takeaways, and Action Items
- Adjustable output length via a token slider
- Option to transcribe only without generating minutes
- Simple Gradio web interface with a public shareable link

---

## Project Structure

```
meeting-minutes-generator/
├── meeting_minutes.ipynb   # Research notebook with step-by-step walkthrough
├── app.py                  # Standalone Gradio application
└── README.md
```

---

## Requirements

- Google Colab with GPU runtime (T4 or better recommended)
- HuggingFace account with access to `meta-llama/Llama-3.2-3B-Instruct`
- HuggingFace token stored as a Colab secret named `HF_TOKEN`

#### Install dependencies

```bash
pip install gradio transformers accelerate bitsandbytes torch huggingface_hub
```

---

## How to Run

#### Option 1 - Run the notebook

Open `meeting_minutes.ipynb` in Google Colab, set your runtime to GPU, and run all cells in order.

#### Option 2 - Run the Gradio app

In Colab, run the following before launching:

```python
import os
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')

!python app.py
```

A public Gradio link will appear in the output. Click it to open the app in your browser.

---

## Output Format

The generated minutes follow this structure:

- **Summary** - Meeting date, location, and list of attendees
- **Discussion Points** - Core topics debated or presented
- **Key Takeaways** - Most significant conclusions reached
- **Action Items** - Table of tasks with assigned owners

---

## Tech Stack

- Python
- HuggingFace Transformers
- Whisper
- Llama 3.2
- BitsAndBytes (4-bit quantization)
- Gradio
- Google Colab
