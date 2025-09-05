import streamlit as st
import joblib, random, json
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect
import datetime

# =============================
# Paths
# =============================
DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
LOG_PATH = Path(__file__).resolve().parent / "data" / "chat_logs.json"
KEYWORDS_PATH = Path(__file__).resolve().parent / "data" / "lang_keywords.json"

# =============================
# Log interactions
# =============================
def log_interaction(user_text, detected_lang, translated_input, predicted_tag, bot_reply):
    """Save each chat interaction into chat_logs.json"""
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "user_text": user_text,
            "detected_lang": detected_lang,
            "translated_input": translated_input,
            "predicted_tag": predicted_tag,
            "bot_reply": bot_reply
        })

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save log: {e}")

# =============================
# Load model & responses
# =============================
clf = joblib.load(MODEL_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ Support both "tag" and "intent"
responses = {}
for intent in data["intents"]:
    tag = intent.get("tag") or intent.get("intent")  # flexible
    responses[tag] = intent.get("responses", [])

# =============================
# Page Configuration
# =============================
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖", layout="wide")

logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=120)
st.title("🎓 University FAQ Chatbot 🤖 (English / Malay / Chinese)")

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This chatbot answers common questions about **university admissions, fees, exams, "
    "library, scholarships, and more.**\n\n"
    "💡 Powered by `scikit-learn`, `Streamlit`, `deep-translator`, and `langdetect`."
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
# Load keyword config
# =============================
if KEYWORDS_PATH.exists():
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        lang_keywords = json.load(f)
else:
    lang_keywords = {"ms": [], "zh-CN": []}

# =============================
# Language detection helper
# =============================
def detect_supported_lang(text):
    t = text.lower()

    if any(word in t for word in lang_keywords.get("ms", [])):
        return "ms"
    if any(word in text for word in lang_keywords.get("zh-CN", [])):
        return "zh-CN"

    try:
        lang = detect(text)
    except:
        return "en"

    if lang in ["en"]:
        return "en"
    elif lang in ["ms", "id"]:
        return "ms"
    elif lang in ["zh", "zh-cn", "zh-tw"]:
        return "zh-CN"
    else:
        return "en"

def lang_label(lang_code):
    if lang_code == "en":
        return "🌍 Detected: English"
    elif lang_code == "ms":
        return "🌍 Detected: Malay"
    elif lang_code == "zh-CN":
        return "🌍 Detected: Chinese"
    return "🌍 Detected: English"

# =============================
# Bot reply
# =============================
def bot_reply(user_text):
    detected_lang = detect_supported_lang(user_text)

    # Translate input if needed
    if detected_lang != "en":
        translated_input = GoogleTranslator(source="auto", target="en").translate(user_text)
    else:
        translated_input = user_text

    # Predict intent
    try:
        tag = clf.predict([translated_input.lower()])[0]
    except Exception:
        tag = "fallback"

    # English response
    reply_en = random.choice(responses.get(tag, responses.get("fallback", ["Sorry, I didn’t understand that."])))

    # Translate back if needed
    if detected_lang != "en":
        reply = GoogleTranslator(source="en", target=detected_lang).translate(reply_en)
    else:
        reply = reply_en

    # Save chat history for UI
    st.session_state.history.append(("You", f"{user_text}\n\n_{lang_label(detected_lang)}_"))
    st.session_state.history.append(("Bot", reply))

    log_interaction(user_text, detected_lang, translated_input, tag, reply)

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
