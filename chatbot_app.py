import streamlit as st
import joblib, random, json
from pathlib import Path
from gtts import gTTS
import speech_recognition as sr
import os

# =============================
# Load model & responses
# =============================
DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
clf = joblib.load(MODEL_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}

# =============================
# Page Configuration
# =============================
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖", layout="wide")

logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=120)
st.title("🎓 University FAQ Chatbot 🤖 with Voice")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This chatbot answers common questions about **university admissions, fees, exams, "
    "library, scholarships, and more.**\n\n"
    "💡 Powered by `scikit-learn`, `Streamlit`, and voice features (`SpeechRecognition`, `gTTS`)."
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
.chat-bubble {
    display: inline-block;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    font-size: 16px;
    word-wrap: break-word;
    max-width: 70%;
    min-width: 50px;
    color: var(--text-color);
}
.user { background-color: #DCF8C6; float: right; clear: both; text-align: right; }
.bot { background-color: #F1F0F0; float: left; clear: both; text-align: left; }
@media (prefers-color-scheme: dark) {
    .bot { background-color: #2E2E2E; }
    .user { background-color: #3A523A; }
}
</style>
""", unsafe_allow_html=True)

# =============================
# Session state
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# Helper: bot reply + TTS
# =============================
def bot_reply(user_text):
    try:
        tag = clf.predict([user_text.lower()])[0]
    except Exception:
        tag = "fallback"
    reply = random.choice(responses.get(tag, responses["fallback"]))
    st.session_state.history.append(("You", user_text))
    st.session_state.history.append(("Bot", reply))

    # Generate voice output
    tts = gTTS(reply)
    audio_file = "bot_reply.mp3"
    tts.save(audio_file)
    st.audio(audio_file, format="audio/mp3")

# =============================
# Quick FAQ Buttons
# =============================
st.markdown("### 🔍 Quick Questions")
col1, col2, col3 = st.columns(3)

if col1.button("📚 Admission Requirements"):
    bot_reply("what are the admission requirements")

if col2.button("💰 Tuition Fees"):
    bot_reply("how much is the tuition fee")

if col3.button("📅 Exam Dates"):
    bot_reply("when are the exams")

# =============================
# Voice Input
# =============================
if st.button("🎤 Speak Your Question"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... please speak clearly")
        audio = recognizer.listen(source)
        try:
            spoken_text = recognizer.recognize_google(audio)
            st.success(f"You said: {spoken_text}")
            bot_reply(spoken_text)
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand your speech.")
        except sr.RequestError:
            st.error("⚠️ Speech recognition service unavailable.")

# =============================
# Text Input
# =============================
if user_input := st.chat_input("Ask me anything about the university..."):
    bot_reply(user_input)

# =============================
# Display chat history
# =============================
for speaker, msg in st.session_state.history:
    bubble_class = "user" if speaker == "You" else "bot"
    prefix = "🧑" if speaker == "You" else "🤖"
    st.markdown(
        f'<div class="chat-bubble {bubble_class}">{prefix} {msg}</div>',
        unsafe_allow_html=True
    )
