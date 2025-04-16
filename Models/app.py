import os
import streamlit as st
import openai
import torch
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from news_video import VideoGenerator
from dotenv import load_dotenv

# === SETUP ===
load_dotenv()
video_api_key = os.getenv("BEARER_TOKEN")
video_generator = VideoGenerator(video_api_key)

st.set_page_config(page_title="AI News Anchor", layout="wide")

# === CUSTOM CSS ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background-color: #000000;
        color: #FFD700;
    }

    .block-container {
        padding: 2rem 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #FFD700 !important;
        text-align: center;
    }

    .stButton>button {
        background-color: #FFD700;
        color: black;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75em 1em;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px #FFD70088;
    }

    .stButton>button:hover {
        background-color: black;
        color: #FFD700;
        border: 1px solid #FFD700;
        box-shadow: 0 0 20px #FFD700;
        cursor: pointer;
    }

    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #FFD700;
        border-radius: 8px;
        border: 1px solid #FFD70044;
    }

    .stTextInput label {
        color: #FFD700;
        font-weight: bold;
    }

    .stAlert {
        background-color: #222222;
        border-left: 5px solid #FFD700;
    }

    .stImage>img {
        border: 2px solid #FFD700;
        border-radius: 12px;
        box-shadow: 0 0 15px #FFD70055;
    }

    .stVideo video {
        border: 2px solid #FFD700;
        border-radius: 12px;
        box-shadow: 0 0 15px #FFD70055;
    }

    .css-1d391kg {
        background-color: #000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# === HEADER ===
st.title("🧠 AI AVATAR")
st.subheader("AUTOMATED VIDEO GENERATION WITH AI AVATAR")

user_image_url = st.text_input(
    "Enter public URL of your image (optional)",
    value="https://i.ibb.co/hYcxXTW/anchor.png"
)
image_url = user_image_url

# === LOAD WHISPER ===
@st.cache_resource
def load_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-tiny"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=0 if torch.cuda.is_available() else -1,
        model_kwargs={"language": "en"}
    )
    return pipe

pipe = load_whisper()

# === AUDIO RECORDING ===
def record_audio(duration=10, sample_rate=16000):
    st.info("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording complete!")
    return np.squeeze(audio), sample_rate

# === RECORDING UI ===
st.write("🎬 Click below to record your news script:")

if st.button("🎤 Record Audio and Generate News Video"):
    audio_array, sample_rate = record_audio()

    st.write("📝 Transcribing your voice...")
    sample = {"array": audio_array, "sampling_rate": sample_rate}
    result = pipe(sample, return_timestamps=True)
    transcript = result["text"]
    st.success("Transcription Complete")

    st.markdown(f"""<div style="background-color:#1a1a1a;padding:15px;border-radius:10px;border:1px solid #FFD700;">
                    <h4 style="color:#FFD700;">📜 Transcribed Text:</h4>
                    <p>{transcript}</p></div>""", unsafe_allow_html=True)

    final_text = f"""
        Hello World, I'm your AI Avatar:
        {transcript}
        That's all for today. Thank you!
    """

    # === SHOW LAYOUT ===
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("🧍 Your AI Avatar")
        st.image(image_url, caption="Avatar Image", use_container_width=True)
    with col2:
        st.warning("⚙️ Generating AI Avatar Video...")
        video_url = video_generator.generate_video(final_text, image_url)
        st.video(video_url)


