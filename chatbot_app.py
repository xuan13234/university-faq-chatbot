import streamlit as st
import joblib, random, json
from pathlib import Path

# Load model & responses
DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
clf = joblib.load(MODEL_PATH)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}

# Page config
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖")
st.title("🎓 University FAQ Chatbot 🤖")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This chatbot answers common questions about **university admissions, fees, exams, "
    "library, scholarships, and more.**\n\n"
    "💡 Powered by `scikit-learn` + `Streamlit`."
)

# Custom CSS for chat bubbles
st.markdown("""
<style>
.chat-bubble {
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    max-width: 70%;
    font-size: 16px;
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
</style>
""", unsafe_allow_html=True)

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input (Streamlit native chat input)
if user_input := st.chat_input("Ask me anything about the university..."):
    try:
        tag = clf.predict([user_input.lower()])[0]
    except Exception:
        tag = "fallback"
    reply = random.choice(responses.get(tag, responses["fallback"]))
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

# Display chat history
for speaker, msg in st.session_state.history:
    bubble_class = "user" if speaker == "You" else "bot"
    prefix = "🧑" if speaker == "You" else "🤖"
    st.markdown(
        f'<div class="chat-bubble {bubble_class}">{prefix} {msg}</div>',
        unsafe_allow_html=True
    )
