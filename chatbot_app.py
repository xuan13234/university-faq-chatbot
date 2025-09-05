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
# Custom CSS for Chat Bubbles
# =============================
st.markdown("""
<style>
.chat-bubble {
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    font-size: 16px;
    color: var(--text-color); /* Adapt font color to theme */
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
@media (prefers-color-scheme: dark) {
    .chat-bubble {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
}
@media (prefers-color-scheme: light) {
    .chat-bubble {
        background-color: #F1F0F0;
        color: #000000;
    }
}
</style>
""", unsafe_allow_html=True)

# =============================
# Session state
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# Quick FAQ Buttons
# =============================
st.markdown("### 🔍 Quick Questions")
col1, col2, col3 = st.columns(3)
if col1.button("📚 Admission Requirements"):
    st.session_state.history.append(("You", "what are the admission requirements"))
if col2.button("💰 Tuition Fees"):
    st.session_state.history.append(("You", "how much is the tuition fee"))
if col3.button("📅 Exam Dates"):
    st.session_state.history.append(("You", "when are the exams"))

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
