import streamlit as st
import torch
import random
import json
from datetime import datetime
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# Load intents
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# Load trained model
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

# Special commands (simulate booking, recommendation, troubleshooting)
def special_commands(msg):
    if msg.startswith("/book"):
        booking_item = msg.split(maxsplit=1)[1] if len(msg.split()) > 1 else "General Service"
        return f"✅ Booking confirmed for {booking_item}."
    elif msg.startswith("/recommend"):
        return "📌 Recommendation: Customers often buy our premium package with extended warranty."
    elif msg.startswith("/troubleshoot"):
        return "🛠️ Try restarting the device. If the issue persists, contact support."
    return None

# Response function
def get_response(msg):
    msg = msg.lower()

    # Handle special commands
    sc = special_commands(msg)
    if sc:
        return sc

    # Preprocess
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
                return random.choice(intent["responses"])
    return "🤔 Sorry, I didn’t quite understand. Could you rephrase?"

# --- Streamlit UI ---
st.set_page_config(page_title="Sales Chatbot", page_icon="🤖")
st.title("💬 Sales Chatbot (Streamlit)")
st.write("Ask me anything about sales, orders, or try `/book`, `/recommend`, `/troubleshoot`")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    response = get_response(user_input)
    st.session_state["messages"].append(("You", user_input))
    st.session_state["messages"].append(("Bot", response))

# Display messages
for sender, msg in st.session_state["messages"]:
    if sender == "You":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
