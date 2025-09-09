import streamlit as st
import torch
import random
import json
import requests
from datetime import datetime
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

# Weather API Key (replace with your own)
WEATHER_API_KEY = "e997151541c24061b4d123258251107"

# Load intents and model
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

# Weather function
def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&lang=en"
    try:
        response = requests.get(url)
        data = response.json()
        if "error" in data:
            return "❌ City not found or API error."
        location = data["location"]["name"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        return f"🌤️ {location}: {temp_c}°C, {condition}"
    except:
        return "❌ Failed to fetch weather. Check network connection."

# Get response
def get_response(msg):
    msg = msg.lower()

    if msg.startswith("/weather"):
        parts = msg.split(maxsplit=1)
        if len(parts) < 2:
            return "❗ Format: /weather <city>"
        return get_weather(parts[1])

    if "time" in msg or "date" in msg:
        return "🕒 Current time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run through model
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])

    return "🤔 Sorry, I don’t quite understand..."

# --- Streamlit UI ---
st.title("💬 Smart Sales Chatbot")
st.write("Ask me anything about sales, orders, or try `/weather <city>`")

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
