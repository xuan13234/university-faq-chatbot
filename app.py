import streamlit as st
import torch
import random
import json
import pandas as pd
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import DeepChatbot
from nltk_utils import tokenize, stem

# ------------------------
# Load intents & model
# ------------------------
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

data = torch.load("data.pth")

model = DeepChatbot(
    data["vocab_size"],
    data["embed_dim"],
    data["hidden_size"],
    len(data["tags"])
)
model.load_state_dict(data["model_state"])
model.eval()

word2idx = data["word2idx"]
tags = data["tags"]

# ------------------------
# Encode sentences
# ------------------------
def encode_sentence(tokens, max_len=10):
    ids = [word2idx.get(stem(w), 0) for w in tokens]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return ids[:max_len]

# ------------------------
# CSV Logging
# ------------------------
LOG_FILE = "chatbot_logs.csv"
HISTORY_FILE = "chat_history.csv"

for file in [LOG_FILE, HISTORY_FILE]:
    if not os.path.exists(file):
        with open(file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            if file == LOG_FILE:
                writer.writerow(["timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"])
            else:
                writer.writerow(["timestamp", "speaker", "message"])

def log_interaction(user_input, predicted_tag, response, correct=None, feedback=None):
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

def log_history(speaker, message):
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speaker, message])

# ------------------------
# Extra Functions
# ------------------------
def check_faq(msg):
    try:
        with open("faq.csv", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row["question"].lower() in msg.lower():
                    return row["answer"]
    except FileNotFoundError:
        return None
    return None

custom_keywords = {
    "delivery": "faq_shipping",
    "return": "faq_refund",
    "payment": "faq_payment",
    "order": "faq_order"
}

def special_commands(msg):
    if msg.startswith("/book"):
        booking_item = msg.split(maxsplit=1)[1] if len(msg.split()) > 1 else "General Service"
        return f"✅ Booking confirmed for {booking_item}.", "booking"
    elif msg.startswith("/recommend"):
        return "📌 Recommendation: Customers often buy our premium package with extended warranty.", "recommendation"
    elif msg.startswith("/troubleshoot"):
        return "🛠️ Try restarting the device. If the issue persists, contact support.", "troubleshooting"
    return None, None

# ------------------------
# Response Logic
# ------------------------
def get_response(msg):
    msg = msg.lower()

    sc, sc_tag = special_commands(msg)
    if sc:
        return sc, sc_tag

    if "time" in msg or "date" in msg:
        return "🕒 Current time is " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "time_date"

    faq_answer = check_faq(msg)
    if faq_answer:
        return "📚 " + faq_answer, "faq"

    for keyword in custom_keywords:
        if keyword in msg:
            return f"🔍 You are asking about {keyword}. Please visit our help page!", keyword

    tokens = tokenize(msg)
    X = torch.tensor([encode_sentence(tokens)], dtype=torch.long)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.7:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"]), tag

    return "🤔 Sorry, I didn’t quite understand. Could you rephrase?", "unknown"

# ------------------------
# Evaluation Metrics
# ------------------------
def evaluate_chatbot():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        return None

    df = pd.read_csv(LOG_FILE)
    df = df.dropna(subset=["correct"])
    if df.empty:
        return None

    y_true = df["correct"].astype(int)
    y_pred = [1 if c == 1 else 0 for c in y_true]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics, df

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="Deep Learning Chatbot", page_icon="🤖", layout="wide")

# Sidebar Branding
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.title("🤖 Smart Chatbot")
st.sidebar.info("Try `/book`, `/recommend`, `/troubleshoot`.\nAlso ask FAQs like 'How to apply?'.")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "rb") as f:
        st.sidebar.download_button("⬇️ Download Chat History", f, "chat_history.csv")
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "rb") as f:
        st.sidebar.download_button("⬇️ Download Evaluation Logs", f, "chatbot_logs.csv")

st.title("🎓 University FAQ Chatbot (Deep Learning)")
st.markdown("<hr>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chatbot", "📊 Evaluation", "📜 Chat History", "⭐ Rating"])

# --- Chatbot Tab ---
with tab1:
    st.subheader("💬 Start Chatting")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.chat_input("Type your message here...")

    if user_input:
        response, predicted_tag = get_response(user_input)
        st.session_state["messages"].append(("You", user_input))
        st.session_state["messages"].append(("Bot", response, predicted_tag))
        log_history("User", user_input)
        log_history("Bot", response)

    for i, msg in enumerate(st.session_state["messages"]):
        if msg[0] == "You":
            st.chat_message("user").markdown(f"<div style='background:#e6f0ff;padding:8px;border-radius:10px;'>🧑 {msg[1]}</div>", unsafe_allow_html=True)
        else:
            response, predicted_tag = msg[1], msg[2]
            st.chat_message("assistant").markdown(f"<div style='background:#f2f2f2;padding:8px;border-radius:10px;'>🤖 {response}</div>", unsafe_allow_html=True)

            if i == len(st.session_state["messages"]) - 1:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍", key=f"yes_{hash(response)}"):
                        log_interaction(st.session_state["messages"][-2][1], predicted_tag, response, 1, "yes")
                        st.success("Feedback saved!")
                with col2:
                    if st.button("👎", key=f"no_{hash(response)}"):
                        log_interaction(st.session_state["messages"][-2][1], predicted_tag, response, 0, "no")
                        st.error("Feedback saved!")

# --- Evaluation Tab ---
with tab2:
    st.subheader("📊 Evaluation Results")
    results = evaluate_chatbot()
    if results:
        metrics, df = results
        st.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
        st.metric("Precision", f"{metrics['Precision']:.2f}")
        st.metric("Recall", f"{metrics['Recall']:.2f}")
        st.metric("F1 Score", f"{metrics['F1 Score']:.2f}")

        st.write("### Feedback by Intent")
        feedback_summary = df.groupby(["predicted_tag", "feedback"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        feedback_summary.plot(kind="bar", ax=ax, color=["green", "red"], width=0.6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
        plt.ylabel("Count")
        plt.title("👍 vs 👎 per Intent", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No evaluation data yet.")

# --- Chat History Tab ---
with tab3:
    st.subheader("📜 Conversation History")
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        history_df = pd.read_csv(HISTORY_FILE)
        st.dataframe(history_df.tail(50))
    else:
        st.info("No chat history yet.")

# --- Rating Tab ---
with tab4:
    st.subheader("⭐ Rate Your Experience")
    rating = st.slider("How would you rate the chatbot?", 1, 5, 3)
    if st.button("Submit Rating"):
        with open("ratings.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating])
        st.success(f"Thanks for rating us {rating} ⭐!")
