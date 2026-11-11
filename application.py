import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import ollama
from faster_whisper import WhisperModel


# Application configuration
APP_DIR = Path(__file__).resolve().parent
DEFAULT_LLM = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
PIPER_MODEL = Path(os.getenv("PIPER_MODEL", APP_DIR / "models" / "en_US-lessac-medium.onnx"))

# System prompt – Chatbot behaviour
SYSTEM_PROMPT = """You are an intelligent, friendly AI Voice Assistant.
Understand the user's questions and provide accurate, relevant answers.
Use simple, natural conversational English.
Keep answers concise and suitable for spoken delivery unless more detail is requested.
Use previous conversation turns when relevant.
Ask a clarifying question when the request is ambiguous.
Do not invent facts or claim to have performed actions you did not perform.
"""

# Streamlit page configuration
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️", layout="centered")


# Loading the Whisper model
@st.cache_resource(show_spinner="Loading speech recognition model…")
def load_whisper(model_name: str):
    # CPU + INT8 only, no dedicated GPU.
    return WhisperModel(model_name, device="cpu", compute_type="int8")


# Speech-to-Text function 
def transcribe_audio(audio_bytes: bytes, model_name: str) -> str:
    model = load_whisper(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)
        audio_path = audio_file.name
    try:
        segments, _ = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        Path(audio_path).unlink(missing_ok=True)

def ask_ollama(prompt: str, history: list, model_name: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep a bounded conversation window to reduce CPU/RAM use.
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": prompt})
    result = ollama.chat(
        model=model_name,
        messages=messages,
        options={"temperature": 0.4, "num_ctx": 4096},
    )
    return result["message"]["content"].strip()

def synthesize_speech(text: str) -> bytes:
    piper_executable = shutil.which("piper")
    if not piper_executable:
        raise RuntimeError(
            "Piper CLI was not found. Install piper-tts in the active virtual environment "
            "and reopen the terminal."
        )
    if not PIPER_MODEL.exists():
        raise FileNotFoundError(
            f"Piper voice model not found: {PIPER_MODEL}. Download the .onnx voice model "
            "and its matching .onnx.json file; see README.md."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
        output_path = Path(output_file.name)
    try:
        completed = subprocess.run(
            [piper_executable, "--model", str(PIPER_MODEL), "--output_file", str(output_path)],
            input=text,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Piper could not generate audio. {detail}")
        return output_path.read_bytes()
    finally:
        output_path.unlink(missing_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio" not in st.session_state:
    st.session_state.last_audio = {}

st.title("🎙️ AI Voice Assistant")
st.caption("A private, locally running voice chatbot powered by open-source AI.")

with st.sidebar:
    st.subheader("Local model settings")
    llm_model = st.text_input("Ollama model", value=DEFAULT_LLM)
    whisper_model_name = st.selectbox(
        "Speech recognition model",
        ["tiny", "base", "small"],
        index=["tiny", "base", "small"].index(WHISPER_MODEL_NAME)
        if WHISPER_MODEL_NAME in ["tiny", "base", "small"] else 1,
    )
    st.caption("For 16 GB RAM and CPU-only use, start with Llama 3.2 3B and Whisper base or tiny.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_audio = {}
        st.rerun()

st.markdown("### Conversation")
if not st.session_state.messages:
    st.info("Hello! Ask a question by typing below or recording your voice.")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and index in st.session_state.last_audio:
            st.audio(st.session_state.last_audio[index], format="audio/wav")

# Typed input
typed_prompt = st.chat_input("Type your message…")
if typed_prompt:
    st.session_state.messages.append({"role": "user", "content": typed_prompt})
    try:
        with st.spinner("Thinking locally…"):
            answer = ask_ollama(typed_prompt, st.session_state.messages[:-1], llm_model)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as exc:
        st.error(f"Could not get a response from Ollama: {exc}")
    st.rerun()

st.divider()
st.markdown("### Voice input")
recording = st.audio_input("Record your question")
if recording is not None:
    st.audio(recording, format="audio/wav")
    if st.button("Transcribe and send voice message", type="primary"):
        try:
            with st.spinner("Transcribing your speech…"):
                spoken_text = transcribe_audio(recording.getvalue(), whisper_model_name)
            if not spoken_text:
                st.warning("No speech was detected. Please try recording again.")
            else:
                st.markdown(f"**Recognized speech:** {spoken_text}")
                st.session_state.messages.append({"role": "user", "content": spoken_text})
                with st.spinner("Generating your answer…"):
                    answer = ask_ollama(
                        spoken_text, st.session_state.messages[:-1], llm_model
                    )
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
        except Exception as exc:
            st.error(f"Voice processing failed: {exc}")

# TTS controls for the most recent assistant response
latest_assistant = next(
    (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
    None,
)
if latest_assistant:
    st.divider()
    st.markdown("### Hear the latest answer")
    if st.button("🔊 Generate voice response"):
        try:
            with st.spinner("Generating speech…"):
                audio_bytes = synthesize_speech(latest_assistant["content"])
            # Store audio separately so it remains available after reruns.
            assistant_indices = [
                i for i, m in enumerate(st.session_state.messages)
                if m["role"] == "assistant"
            ]
            st.session_state.last_audio[assistant_indices[-1]] = audio_bytes
            st.audio(audio_bytes, format="audio/wav")
        except Exception as exc:
            st.error(f"Text-to-speech failed: {exc}")

st.caption("Audio recordings are processed locally by this application. This prototype does not save recordings to a database.")
