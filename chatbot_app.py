import streamlit as st
import joblib, random, json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

clf = joblib.load(MODEL_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}

st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖")
st.title("🎓 University FAQ Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Type your question:", "")

if user_input:
    try:
        tag = clf.predict([user_input.lower()])[0]
    except Exception:
        tag = "fallback"
    reply = random.choice(responses.get(tag, responses["fallback"]))
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")