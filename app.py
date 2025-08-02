import os
import sys
import urllib.request
import zipfile

def download_and_extract_vosk_model():
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    model_zip = "vosk-model-small-en-us-0.15.zip"
    model_dir = "vosk-model-small-en-us-0.15"

    if not os.path.exists(model_dir):
        st.info("Downloading Vosk model (~50MB), please wait...")
        urllib.request.urlretrieve(model_url, model_zip)
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(model_zip)
        st.success("Vosk model downloaded and extracted!")

vosk_model_path = "vosk-model-small-en-us-0.15"

if not os.path.exists(vosk_model_path):
    download_and_extract_vosk_model()

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import kagglehub
import torch
from PIL import Image
import pyttsx3
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import os

# --- Vosk Speech-to-Text Setup ---
def record_and_transcribe():
    st.info("🎤 Listening... Speak into your microphone.")
    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    vosk_model_path = "vosk-model-small-en-us-0.15"
    if not os.path.exists(vosk_model_path):
        st.error("VOSK model not found! Download it from https://alphacephei.com/vosk/models and extract here.")
        return ""

    model = Model(vosk_model_path)
    rec = KaldiRecognizer(model, 16000)

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        full_text = ""
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                full_text = result.get("text", "")
                break
        return full_text

# --- Load Gemma Model ---
@st.cache_resource
def load_gemma():
    model_path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e4b-it")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

def gemma_response(prompt):
    tokenizer, model = load_gemma()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- Text to Speech ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# --- Streamlit UI ---
st.title("🌾 Farmer's Leaf Assistant (Offline)")
st.write("Speak your question and upload a leaf photo. This app works fully offline.")

if st.button("🎙️ Record Your Question"):
    try:
        user_text = record_and_transcribe()
        st.success(f"🗣️ You said: '{user_text}'")
    except Exception as e:
        st.error(f"Voice error: {str(e)}")
        user_text = ""

uploaded_image = st.file_uploader("📸 Upload a photo of the leaf", type=["jpg", "jpeg", "png"])
if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Leaf", use_column_width=True)

if st.button("🧠 Analyze and Get Advice"):
    if not uploaded_image:
        st.warning("Please upload a leaf image.")
    elif 'user_text' not in locals() or not user_text:
        st.warning("Please record a question.")
    else:
        prompt = f"A farmer asked: '{user_text}' while showing a leaf. Provide simple farming advice."
        response = gemma_response(prompt)
        st.success("💬 AI Response:")
        st.text_area("Response:", response, height=200)
        speak(response)
