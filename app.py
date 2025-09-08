import streamlit as st
import torch
import random
import json
import pandas as pd
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from model import DeepChatbot
from nltk_utils import tokenize, stem
from collections import defaultdict

# Optional imports (translation & detection)
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except Exception:
    HAS_LANGDETECT = False

try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    HAS_DEEP_TRANSLATOR = True
except Exception:
    HAS_DEEP_TRANSLATOR = False

try:
    from googletrans import Translator as GoogleTransTranslator
    HAS_GOOGLETRANS = True
except Exception:
    HAS_GOOGLETRANS = False

# ------------------------
# Config
# ------------------------
LOG_FILE = "chatbot_logs.csv"
HISTORY_FILE = "chat_history.csv"
FAQ_FILE = "faq.csv"
MAX_LEN = 16
PROB_THRESHOLD = 0.70

# ------------------------
# Ensure CSV
# ------------------------
def ensure_csv(file_path, header):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(header)

ensure_csv(LOG_FILE, ["timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"])
ensure_csv(HISTORY_FILE, ["timestamp", "speaker", "message"])

# ------------------------
# Load resources
# ------------------------
@st.cache_resource
def load_resources():
    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    data = torch.load("data.pth", map_location=torch.device('cpu'))

    model = DeepChatbot(
        data["vocab_size"],
        data["embed_dim"],
        data["hidden_size"],
        len(data["tags"])
    )
    model.load_state_dict(data["model_state"])
    model.eval()

    return intents, data, model

try:
    intents, data, model = load_resources()
    word2idx = data.get("word2idx", {})
    tags = data.get("tags", [])
except Exception as e:
    st.error(f"Failed to load model: {e}")
    raise

# ------------------------
# Encoding
# ------------------------
UNK_IDX = word2idx.get("<UNK>", 0)

def encode_sentence(tokens, max_len=MAX_LEN):
    ids = [word2idx.get(stem(w), UNK_IDX) for w in tokens]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids[:max_len]

# ------------------------
# Logging
# ------------------------
def log_interaction(user_input, predicted_tag, response, correct=None, feedback=None):
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_input,
                predicted_tag,
                response,
                correct,
                feedback
            ])
    except Exception:
        pass

def log_history(speaker, message):
    try:
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speaker, message])
    except Exception:
        pass

# ------------------------
# Translation
# ------------------------
def detect_language_safe(text):
    if not HAS_LANGDETECT or not text:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_to_en(text, src=None):
    if not text:
        return text, src
    if HAS_DEEP_TRANSLATOR:
        try:
            res = DeepGoogleTranslator(source=src if src else "auto", target="en").translate(text)
            return res, src
        except Exception:
            pass
    if HAS_GOOGLETRANS:
        try:
            t = GoogleTransTranslator()
            res = t.translate(text, dest="en")
            return res.text, res.src
        except Exception:
            pass
    return text, src

def translate_from_en(text, target):
    if not text or not target or target == "en":
        return text
    if HAS_DEEP_TRANSLATOR:
        try:
            return DeepGoogleTranslator(source="en", target=target).translate(text)
        except Exception:
            pass
    if HAS_GOOGLETRANS:
        try:
            t = GoogleTransTranslator()
            res = t.translate(text, dest=target)
            return res.text
        except Exception:
            pass
    return text

# ------------------------
# FAQ + commands
# ------------------------
def check_faq(msg):
    if not os.path.exists(FAQ_FILE):
        return None
    try:
        with open(FAQ_FILE, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("question") and row["question"].lower() in msg.lower():
                    return row.get("answer")
    except Exception:
        return None
    return None

def special_commands(msg):
    if msg.startswith("/book"):
        return f"✅ Booking confirmed for {msg[6:] if len(msg.split()) > 1 else 'General Service'}.", "booking"
    elif msg.startswith("/recommend"):
        return "📌 Recommendation: Try our premium package with extended warranty.", "recommendation"
    elif msg.startswith("/troubleshoot"):
        return "🛠️ Try restarting the device. If the issue persists, contact support.", "troubleshooting"
    return None, None

# ------------------------
# Core response
# ------------------------
def classify_message(msg):
    tokens = tokenize(msg)
    X = torch.tensor([encode_sentence(tokens)], dtype=torch.long)
    with torch.no_grad():
        output = model(X)
        probs = torch.softmax(output, dim=1)
        top_prob, predicted = torch.max(probs, dim=1)
        tag = tags[predicted.item()]
    return tag, top_prob.item()

def get_response(msg):
    if not msg.strip():
        return "🤔 Please type something.", "empty"

    sc, sc_tag = special_commands(msg)
    if sc:
        return sc, sc_tag

    if "time" in msg or "date" in msg:
        return "🕒 Current time is " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "time_date"

    faq_answer = check_faq(msg)
    if faq_answer:
        return "📚 " + faq_answer, "faq"

    tag, prob = classify_message(msg)
    if prob >= PROB_THRESHOLD:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent.get("responses", ["I can help with that."])), tag
    return "🤔 Sorry, I didn’t quite understand.", "unknown"

# ------------------------
# Evaluation
# ------------------------
def evaluate_chatbot():
    if not os.path.exists(LOG_FILE):
        return None
    try:
        df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
        if df.empty:
            return None
        return df
    except Exception:
        return None

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Deep Learning Chatbot", page_icon="🤖", layout="wide")

st.sidebar.title("🤖 Chatbot Control")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "rb") as f:
        st.sidebar.download_button("⬇️ Download Chat History", f, "chat_history.csv")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as f:
        st.sidebar.download_button("⬇️ Download Logs", f, "chatbot_logs.csv")

st.title("🎓 University FAQ Chatbot")
tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.chat_input("Type your message...")

    if user_input:
        lang = detect_language_safe(user_input)
        translated_input, _ = translate_to_en(user_input, src=lang)
        response, tag = get_response(translated_input)
        final_response = translate_from_en(response, lang)

        st.session_state["messages"].append(("You", user_input))
        st.session_state["messages"].append(("Bot", final_response))

        log_history("User", user_input)
        log_history("Bot", final_response)
        log_interaction(user_input, tag, final_response)

    for i, (speaker, text) in enumerate(st.session_state["messages"]):
        if speaker == "You":
            st.chat_message("user").markdown(f"🧑 {text}")
        else:
            st.chat_message("assistant").markdown(f"🤖 {text}")

with tab2:
    st.subheader("📊 Evaluation")
    df = evaluate_chatbot()
    if df is not None and not df.empty:
        st.write(df.tail(20))
        if "predicted_tag" in df.columns:
            summary = df.groupby("predicted_tag").size().reset_index(name="count")
            st.dataframe(summary)
            fig, ax = plt.subplots()
            ax.bar(summary["predicted_tag"], summary["count"])
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)
    else:
        st.info("No logs yet.")
