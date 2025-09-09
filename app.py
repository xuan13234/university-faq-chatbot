import os
import csv
import json
import random
import re
from datetime import datetime
from collections import deque
import time

import streamlit as st
import pandas as pd
import numpy as np

# Set page config first
st.set_page_config(
    page_title="Advanced Deep NLP Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create data directory if it doesn't exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Custom CSS for styling
def inject_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .user-message {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .bot-message {
        background: linear-gradient(135deg, #e0e0e0, #f5f5f5);
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .message-meta {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 4px;
    }
    .stButton button {
        border-radius: 20px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .debug-info {
        background-color: #f8f9fa;
        border-left: 4px solid #6e8efb;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# Load intents
def load_intents():
    try:
        if os.path.exists(os.path.join(DATA_DIR, "intents.json")):
            with open(os.path.join(DATA_DIR, "intents.json"), "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Create default intents if file doesn't exist
            default_intents = {
                "intents": [
                    {
                        "tag": "greeting",
                        "patterns": ["hello", "hi", "hey", "how are you", "good day", "what's up"],
                        "responses": ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Hey! How can I assist you?"]
                    },
                    {
                        "tag": "goodbye",
                        "patterns": ["bye", "see you later", "goodbye", "take care", "see ya"],
                        "responses": ["Goodbye! Have a great day!", "See you later!", "Take care!"]
                    },
                    {
                        "tag": "fees",
                        "patterns": [
                            "what are the fees", "how much does it cost", "what is the course fee", 
                            "tell me about pricing", "what's the cost", "fee structure",
                            "payment information", "how much do I need to pay", "tuition fees",
                            "course pricing", "how much is the course", "what are the costs"
                        ],
                        "responses": [
                            "Our course fees vary depending on the program. Could you specify which course you're interested in?",
                            "The fee structure is available on our website. Would you like me to direct you to the fees page?",
                            "For detailed information about course fees, please contact our admissions office at admissions@example.com.",
                            "We offer various payment plans. The standard course fee is $X, but it may vary by program."
                        ]
                    },
                    {
                        "tag": "courses",
                        "patterns": [
                            "what courses do you offer", "tell me about your programs", "available courses",
                            "what programs are available", "list of courses", "degree programs",
                            "what can I study", "educational programs", "curriculum options",
                            "learning paths", "what do you teach", "what subjects are available"
                        ],
                        "responses": [
                            "We offer a wide range of courses in various fields. Could you specify your area of interest?",
                            "Our programs include Computer Science, Business Administration, Engineering, and more. Which field are you interested in?",
                            "You can view our complete course catalog on our website. Would you like me to direct you there?",
                            "We offer undergraduate, graduate, and certificate programs across multiple disciplines."
                        ]
                    },
                    {
                        "tag": "thanks",
                        "patterns": ["thank you", "thanks", "appreciate it", "thank you very much"],
                        "responses": ["You're welcome!", "Happy to help!", "Anytime!", "Glad I could assist you!"]
                    },
                    {
                        "tag": "hours",
                        "patterns": ["what are your hours", "when are you open", "what time do you open", "when do you close"],
                        "responses": ["We're open from 9 AM to 5 PM, Monday to Friday.", "Our hours are 9 AM to 5 PM on weekdays."]
                    },
                    {
                        "tag": "contact",
                        "patterns": ["how can I contact you", "what's your phone number", "where are you located", "email address"],
                        "responses": ["You can reach us at contact@example.com or call 555-1234.", "We're located at 123 Main Street. Phone: 555-1234."]
                    }
                ]
            }
            with open(os.path.join(DATA_DIR, "intents.json"), "w", encoding="utf-8") as f:
                json.dump(default_intents, f, indent=2)
            return default_intents
    except Exception as e:
        st.sidebar.error(f"Error loading intents: {e}")
        return {"intents": []}

intents = load_intents()

# Text cleaning function
def clean_text(text):
    if text is None:
        return ""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# Time-related questions handler
def handle_time_question(text):
    text_lower = text.lower()
    
    if any(phrase in text_lower for phrase in ["time", "what time", "current time", "what is the time"]):
        return f"🕒 The current time is {datetime.now().strftime('%H:%M:%S')}"
    
    if any(phrase in text_lower for phrase in ["date", "what date", "current date", "what is the date"]):
        return f"📅 Today's date is {datetime.now().strftime('%Y-%m-%d')}"
    
    if any(phrase in text_lower for phrase in ["day", "what day", "today", "what day is it"]):
        return f"📆 Today is {datetime.now().strftime('%A')}"
    
    return None

# Special commands handler
def special_commands(msg):
    if not msg:
        return None
    
    msg_lower = msg.lower()
    
    if msg_lower.startswith("/book"):
        parts = msg.split(maxsplit=1)
        item = parts[1] if len(parts) > 1 else "General Service"
        return ("booking", f"✅ Booking confirmed for {item}. We will contact you.")
    
    if msg_lower.startswith("/recommend"):
        return ("recommendation", "📌 Recommendation: Premium plan + warranty.")
    
    if msg_lower.startswith("/troubleshoot"):
        return ("troubleshoot", "🛠️ Try restarting the device; if issue persists, contact support.")
    
    if msg_lower.startswith("/help"):
        help_text = "🤖 Available commands:\n\n"
        help_text += "• /book [item] - Book a service\n"
        help_text += "• /recommend - Get recommendations\n"
        help_text += "• /troubleshoot - Get troubleshooting help\n"
        help_text += "• /clear - Clear chat history\n"
        help_text += "• /feedback - Provide feedback\n\n"
        help_text += "I can also help with these topics:\n"
        
        # Add intents to help text
        for intent in intents.get("intents", []):
            if intent.get("patterns"):
                help_text += f"• {intent['patterns'][0]}\n"
        
        return ("help", help_text)
    
    if msg_lower.startswith("/clear"):
        st.session_state.messages = []
        st.session_state.context = deque(maxlen=5)
        return ("clear", "🗑️ Chat history cleared.")
    
    if msg_lower.startswith("/feedback"):
        parts = msg.split(maxsplit=1)
        feedback = parts[1] if len(parts) > 1 else ""
        if feedback:
            with open(os.path.join(DATA_DIR, "user_feedback.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {feedback}\n")
            return ("feedback", "📝 Thank you for your feedback!")
        else:
            return ("feedback", "📝 Please provide your feedback after the /feedback command.")
    
    return None

# Improved intent matching
def match_intent(user_input):
    # First check for special commands
    sc = special_commands(user_input)
    if sc:
        return sc[0], sc[1], 1.0
    
    # Check for time-related questions
    time_response = handle_time_question(user_input)
    if time_response:
        return "time", time_response, 1.0
    
    cleaned_input = clean_text(user_input)
    
    # Check for exact matches first
    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            cleaned_pattern = clean_text(pattern)
            if cleaned_pattern == cleaned_input:
                return intent.get("tag"), random.choice(intent.get("responses", [])), 1.0
    
    # Check for partial matches
    best_match = None
    best_score = 0
    best_response = None
    
    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            cleaned_pattern = clean_text(pattern)
            
            # Check if pattern is in user input
            if cleaned_pattern in cleaned_input:
                score = len(cleaned_pattern) / len(cleaned_input)
                if score > best_score:
                    best_score = score
                    best_match = intent.get("tag")
                    best_response = random.choice(intent.get("responses", []))
            
            # Check if user input is in pattern (for shorter queries)
            if cleaned_input in cleaned_pattern and len(cleaned_input) > 3:
                score = len(cleaned_input) / len(cleaned_pattern)
                if score > best_score:
                    best_score = score
                    best_match = intent.get("tag")
                    best_response = random.choice(intent.get("responses", []))
    
    # If we have a reasonable match, return it
    if best_match and best_score > 0.5:
        return best_match, best_response, best_score
    
    # Check for keyword matches as fallback
    word_scores = {}
    input_words = set(cleaned_input.split())
    
    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            cleaned_pattern = clean_text(pattern)
            pattern_words = set(cleaned_pattern.split())
            
            # Calculate intersection of words
            common_words = input_words.intersection(pattern_words)
            if common_words:
                score = len(common_words) / len(input_words)
                if intent.get("tag") not in word_scores or score > word_scores[intent.get("tag")]:
                    word_scores[intent.get("tag")] = score
    
    # Find the best keyword match
    if word_scores:
        best_tag = max(word_scores, key=word_scores.get)
        if word_scores[best_tag] > 0.3:  # Threshold for keyword matching
            for intent in intents.get("intents", []):
                if intent.get("tag") == best_tag:
                    return best_tag, random.choice(intent.get("responses", [])), word_scores[best_tag]
    
    # Default response if no intent matched
    return "unknown", "I'm not sure how to respond to that. Can you try rephrasing your question?", 0.0

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = deque(maxlen=5)
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Main app
st.title("🤖 Advanced Deep NLP Chatbot")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "📜 History", "⚙️ Settings"])

# Chat tab
with tab1:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                🧑 <b>You</b>: {message["content"]}
                <div class="message-meta">{message.get("timestamp", "")}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                🤖 <b>Bot</b>: {message["content"]}
                <div class="message-meta">Intent: {message.get("intent", "N/A")} • Confidence: {message.get("confidence", 0):.2%} • {message.get("timestamp", "")}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show debug info if enabled
            if st.session_state.debug_mode and message.get("debug_info"):
                debug_info = message.get("debug_info", {})
                debug_text = f"""
                <div class="debug-info">
                    <b>Debug Info:</b><br>
                    Method: {debug_info.get('method', 'N/A')}<br>
                """
                
                if 'model_prediction' in debug_info:
                    debug_text += f"Model Prediction: {debug_info['model_prediction'].get('tag', 'N/A')} ({debug_info['model_prediction'].get('confidence', 0):.2%})<br>"
                
                if 'semantic_match' in debug_info:
                    debug_text += f"Semantic Match: {debug_info['semantic_match'].get('tag', 'N/A')} ({debug_info['semantic_match'].get('confidence', 0):.2%})<br>"
                
                if 'keyword_match' in debug_info:
                    debug_text += f"Keyword Match: {debug_info['keyword_match'].get('tag', 'N/A')} ({debug_info['keyword_match'].get('confidence', 0):.2%})<br>"
                
                debug_text += "</div>"
                st.markdown(debug_text, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        # Get bot response
        tag, response, confidence = match_intent(prompt)
        
        # Add debug info if enabled
        debug_info = {}
        if st.session_state.debug_mode:
            debug_info = {
                "method": "rule_based",
                "input": prompt,
                "cleaned_input": clean_text(prompt),
                "matched_intent": tag,
                "confidence": confidence
            }
        
        # Add bot response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "intent": tag,
            "confidence": confidence,
            "timestamp": timestamp,
            "debug_info": debug_info
        })
        
        # Rerun to update the UI
        st.rerun()

# Analytics tab
with tab2:
    st.header("📊 Chat Analytics")
    
    if st.session_state.messages:
        # Calculate some basic stats
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        bot_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        # Count intents
        intent_counts = {}
        for message in st.session_state.messages:
            if message["role"] == "assistant" and "intent" in message:
                intent = message["intent"]
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("User Messages", user_messages)
        with col3:
            st.metric("Bot Messages", bot_messages)
        
        # Display intent distribution
        st.subheader("Intent Distribution")
        if intent_counts:
            intent_df = pd.DataFrame({
                "Intent": list(intent_counts.keys()),
                "Count": list(intent_counts.values())
            })
            st.bar_chart(intent_df.set_index("Intent"))
        else:
            st.info("No intent data available yet.")
    else:
        st.info("No messages yet. Start chatting to see analytics!")

# History tab
with tab3:
    st.header("📜 Chat History")
    
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You** ({message.get('timestamp', '')}): {message['content']}")
            else:
                st.markdown(f"**Bot** ({message.get('timestamp', '')}): {message['content']}")
                if "intent" in message:
                    st.caption(f"Intent: {message['intent']} • Confidence: {message.get('confidence', 0):.2%}")
            st.divider()
        
        # Export button
        if st.button("Export Chat History"):
            history_df = pd.DataFrame(st.session_state.messages)
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="chat_history.csv",
                mime="text/csv"
            )
    else:
        st.info("No chat history yet. Start a conversation!")

# Settings tab
with tab4:
    st.header("⚙️ Settings")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    
    # Intent management
    st.subheader("Intent Management")
    
    # Display current intents
    for intent in intents.get("intents", []):
        with st.expander(f"{intent.get('tag')} ({len(intent.get('patterns', []))} patterns)"):
            st.write("Patterns:")
            for pattern in intent.get("patterns", []):
                st.write(f"- {pattern}")
            st.write("Responses:")
            for response in intent.get("responses", []):
                st.write(f"- {response}")
    
    # Add new intent
    st.subheader("Add New Intent")
    with st.form("add_intent_form"):
        new_tag = st.text_input("Intent Tag")
        new_patterns = st.text_area("Patterns (one per line)")
        new_responses = st.text_area("Responses (one per line)")
        
        if st.form_submit_button("Add Intent"):
            if new_tag and new_patterns:
                # Add new intent to the list
                new_intent = {
                    "tag": new_tag,
                    "patterns": [p.strip() for p in new_patterns.split("\n") if p.strip()],
                    "responses": [r.strip() for r in new_responses.split("\n") if r.strip()] or ["I can help with that."]
                }
                
                intents["intents"].append(new_intent)
                
                # Save to file
                with open(os.path.join(DATA_DIR, "intents.json"), "w", encoding="utf-8") as f:
                    json.dump(intents, f, indent=2)
                
                st.success(f"Added new intent: {new_tag}")
                st.rerun()
    
    # Clear chat history
    st.subheader("Manage Chat")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.context = deque(maxlen=5)
        st.success("Chat history cleared!")
        st.rerun()

# Sidebar
with st.sidebar:
    st.title("🤖 Chatbot Settings")
    st.markdown("---")
    
    st.subheader("Quick Actions")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = deque(maxlen=5)
        st.success("Chat history cleared!")
        st.rerun()
    
    if st.button("View Intents", use_container_width=True):
        st.session_state.debug_mode = True
        st.rerun()
    
    st.markdown("---")
    st.subheader("About This Chatbot")
    st.info("""
    This is an advanced NLP chatbot with intent detection capabilities.
    It can understand and respond to various questions about courses, fees, and more.
    
    Try asking:
    - What courses do you offer?
    - How much does it cost?
    - What are your hours?
    - Thank you
    """)
    
    st.markdown("---")
    st.subheader("System Status")
    st.success("✅ Intent detection: Active")
    st.success("✅ Response generation: Active")
    st.success("✅ Chat history: Enabled")
    
    if st.session_state.debug_mode:
        st.warning("🔧 Debug mode: Enabled")
    else:
        st.info("🔧 Debug mode: Disabled")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Intent detection powered by rule-based matching.")
