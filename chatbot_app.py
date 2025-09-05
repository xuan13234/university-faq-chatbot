import streamlit as st
import joblib, random, json
from pathlib import Path

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

# Add logo (replace with your actual logo in data/university_logo.png)
logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=120)
st.title("🎓 University FAQ Chatbot 🤖")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This chatbot answers common questions about **university admissions, fees, exams, "
    "library, scholarships, and more.**\n\n"
    "💡 Powered by `scikit-learn` + `Streamlit`."
)

# =============================
# Custom CSS (Theme Adaptive)
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
.user {
    background-color: #DCF8C6;
    margin-left: auto;
    text-align: right;
}
.bot {
    background-color: #F1F0F0;
    margin-right: auto;
    text-align: left;
}

/* Dark mode fine-tuning */
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
# Quick FAQ Buttons (with bot response)
# =============================
st.markdown("### 🔍 Quick Questions")
col1, col2, col3 = st.columns(3)

def handle_quick_question(question):
    st.session_state.history.append(("You", question))
    try:
        tag = clf.predict([question.lower()])[0]
    except Exception:
        tag = "fallback"
    reply = random.choice(responses.get(tag, responses["fallback"]))
    st.session_state.history.append(("Bot", reply))

if col1.button("📚 Admission Requirements"):
    handle_quick_question("what are the admission requirements")

if col2.button("💰 Tuition Fees"):
    handle_quick_question("how much is the tuition fee")

if col3.button("📅 Exam Dates"):
    handle_quick_question("when are the exams")

# =============================
# Chat input
# =============================
if user_input := st.chat_input("Ask me anything about the university..."):
    try:
        tag = clf.predict([user_input.lower()])[0]
    except Exception:
        tag = "fallback"
    reply = random.choice(responses.get(tag, responses["fallback"]))
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

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
