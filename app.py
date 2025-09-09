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
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# ------------------------
# Load model and intents
# ------------------------
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

data = torch.load("data.pth")
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# ------------------------
# CSV Logging
# ------------------------
LOG_FILE = "chatbot_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"])

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

# ------------------------
# Response Logic
# ------------------------
def special_commands(msg):
    if msg.startswith("/book"):
        booking_item = msg.split(maxsplit=1)[1] if len(msg.split()) > 1 else "General Service"
        return f"✅ Booking confirmed for {booking_item}.", "booking"
    elif msg.startswith("/recommend"):
        return "📌 Recommendation: Customers often buy our premium package with extended warranty.", "recommendation"
    elif msg.startswith("/troubleshoot"):
        return "🛠️ Try restarting the device. If the issue persists, contact support.", "troubleshooting"
    return None, None

def get_response(msg):
    msg = msg.lower()

    # Special command
    sc, sc_tag = special_commands(msg)
    if sc:
        return sc, sc_tag

    # ML model
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

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

    # Ensure required columns exist
    required_cols = {"timestamp", "user_input", "predicted_tag", "response", "correct", "feedback"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        st.warning(f"⚠️ Missing columns in log file: {missing_cols}. Add some feedback first.")
        return None

    # Drop rows without feedback
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
st.set_page_config(page_title="Sales Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Sales Chatbot with Evaluation")

tab1, tab2 = st.tabs(["💬 Chatbot", "📊 Evaluation"])

# --- Chatbot Tab ---
with tab1:
    st.subheader("Chat with the Bot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.chat_input("Type your message here...")
    if user_input:
        response, predicted_tag = get_response(user_input)
        st.session_state["messages"].append(("You", user_input))
        st.session_state["messages"].append(("Bot", response, predicted_tag))

    for msg in st.session_state["messages"]:
        if msg[0] == "You":
            st.chat_message("user").write(msg[1])
        else:
            response, predicted_tag = msg[1], msg[2]
            st.chat_message("assistant").write(response)

            # Feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"yes_{hash(response)}"):
                    log_interaction(msg[1], predicted_tag, response, 1, "yes")
                    st.success("Feedback saved: Helpful")
            with col2:
                if st.button("👎", key=f"no_{hash(response)}"):
                    log_interaction(msg[1], predicted_tag, response, 0, "no")
                    st.error("Feedback saved: Not helpful")

# --- Evaluation Tab ---
with tab2:
    st.subheader("Chatbot Evaluation Results")
    results = evaluate_chatbot()
    if results:
        metrics, df = results
        st.write("### Metrics")
        st.table({k: f"{v:.2f}" for k, v in metrics.items()})
        st.write("### Logged Interactions")
        st.dataframe(df.tail(10))

        # --- Bar Chart per Intent ---
        st.write("### Feedback by Intent")
        feedback_summary = df.groupby(["predicted_tag", "feedback"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots()
        feedback_summary.plot(kind="bar", ax=ax, color=["green", "red"])
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.title("👍 vs 👎 per Intent")
        st.pyplot(fig)

    else:
        st.info("No feedback data available yet. Chat with the bot and give feedback first!")
