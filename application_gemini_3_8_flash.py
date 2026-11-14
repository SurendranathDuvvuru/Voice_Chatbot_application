import os
import shutil
import subprocess
import tempfile
import time
import logging
import sys
from pathlib import Path
import streamlit as st
from google import genai
from google.genai import types
from faster_whisper import WhisperModel


# ==========================================
# APPLICATION LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

logger = logging.getLogger("VoInCon_AI")

# Application configuration
APP_DIR = Path(__file__).resolve().parent
GEMINI_MODEL_DEFAULT = "gemini-3.8-flash"
try:
    DEFAULT_LLM = st.secrets.get(
        "GEMINI_MODEL",
        os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT),
    )
except Exception:
    DEFAULT_LLM = os.getenv("GEMINI_MODEL", GEMINI_MODEL_DEFAULT)
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

def get_gemini_api_key() -> str:
    """Read the Gemini API key from Streamlit secrets or environment variables."""
    try:
        return st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    except Exception:
        # Supports local execution when no .streamlit/secrets.toml exists.
        return os.getenv("GEMINI_API_KEY", "")


def ask_gemini(prompt: str, history: list, model_name: str) -> str:
    """Generate a response with Gemini and retry temporary server errors."""

    api_key = get_gemini_api_key()

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to Streamlit Cloud Secrets or local environment variables."
        )

    client = genai.Client(api_key=api_key)
    
    # Log user input
    start_time = time.perf_counter()

    logger.info(
        "USER INPUT | Model: %s | Input: %s",
        model_name,
        prompt
    )

    # Prepare conversation history
    contents = []

    for message in history[-12:]:
        role = (
            "model"
            if message.get("role") == "assistant"
            else "user"
        )

        message_text = message.get("content", "")

        if message_text:
            contents.append({
                "role": role,
                "parts": [{"text": message_text}],
            })

    # Add current user prompt
    contents.append({
        "role": "user",
        "parts": [{"text": prompt}],
    })

    # Gemini generation configuration
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
    )

    # Retry configuration
    max_attempts = 3
    retry_delays = [2, 5]

    for attempt in range(max_attempts):

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            answer = (response.text or "").strip()
            # Log AI-generated output
            response_time = time.perf_counter() - start_time

            logger.info(
                "AI OUTPUT | Model: %s | Output: %s",
                model_name,
                answer
                )
            
            logger.info(
                "REQUEST SUCCESS | Response Time: %.2f seconds",
                response_time
                )
            
            if not answer:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return answer

        except Exception as exc:

            error_code = getattr(exc, "code", None)

            # Handle temporary Gemini server errors
            if (
                str(error_code) == "503"
                or "503" in str(exc)
                or "UNAVAILABLE" in str(exc)
            ):

                if attempt < max_attempts - 1:

                    delay = retry_delays[attempt]

                    st.warning(
                        f"Gemini is busy. Retrying in {delay} seconds "
                        f"(attempt {attempt + 2}/{max_attempts})..."
                    )

                    time.sleep(delay)
                    continue

            logger.exception("GEMINI API REQUEST FAILED")
            
            raise RuntimeError(
                f"Gemini API request failed: {exc}"
            ) from exc

    raise RuntimeError(
        "Gemini could not generate a response after multiple attempts."
    )

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
if "chat_error" not in st.session_state:
    st.session_state.chat_error = None
if "tts_notice" not in st.session_state:
    st.session_state.tts_notice = None

st.title("🎙️ AI Voice Assistant")
st.caption("A voice chatbot powered by Gemini, Faster-Whisper and Piper.")

with st.sidebar:
    st.subheader("Gemini and speech settings")
    llm_model = st.text_input("Gemini model", value=DEFAULT_LLM)
    whisper_model_name = st.selectbox(
        "Speech recognition model",
        ["tiny", "base", "small"],
        index=["tiny", "base", "small"].index(WHISPER_MODEL_NAME)
        if WHISPER_MODEL_NAME in ["tiny", "base", "small"] else 1,
    )
    st.caption("Gemini 3.8 Flash is the default cloud model. Whisper base or tiny is a practical CPU starting point.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_audio = {}
        st.session_state.chat_error = None
        st.session_state.tts_notice = None
        st.rerun()

st.markdown("### Conversation")
if not st.session_state.messages:
    st.info("Hello! Ask a question by typing below or recording your voice.")

for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and index in st.session_state.last_audio:
            st.audio(st.session_state.last_audio[index], format="audio/wav")

# Typed input: generate a text answer and automatically synthesize its voice.
typed_prompt = st.chat_input("Type your message…")
if typed_prompt:
    st.session_state.messages.append({"role": "user", "content": typed_prompt})

    try:
        with st.spinner("Generating response with Gemini…"):
            answer = ask_gemini(
                typed_prompt,
                st.session_state.messages[:-1],
                llm_model,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })
        assistant_index = len(st.session_state.messages) - 1

        # Automatically generate speech for the typed-message response.
        try:
            with st.spinner("Generating voice response…"):
                audio_bytes = synthesize_speech(answer)
            st.session_state.last_audio[assistant_index] = audio_bytes
            st.session_state.tts_notice = None
        except Exception as tts_exc:
            # Keep the text answer even if speech generation fails.
            st.session_state.tts_notice = (
                f"Your text response was generated, but voice generation failed: {tts_exc}"
            )

        st.session_state.chat_error = None

    except Exception as exc:
        st.session_state.chat_error = f"Response generation failed: {exc}"

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
                with st.spinner("Generating response with Gemini…"):
                    answer = ask_gemini(spoken_text, st.session_state.messages[:-1], llm_model)
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

st.caption("This prototype does not save recordings to a database.")
