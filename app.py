import os
import csv
import json
import random
import re
from datetime import datetime
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from streamlit.components.v1 import html

# Set a consistent font for matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

# ------------------------
# Optional heavy libraries
# ------------------------
HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
except Exception:
    HAS_TORCH = False
    torch = None
    nn = None

HAS_SBERT = True
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    HAS_SBERT = False
    SentenceTransformer = None
    util = None

HAS_LANGDETECT = True
try:
    from langdetect import detect, DetectorFactory
    # Ensure consistent language detection
    DetectorFactory.seed = 0
except Exception:
    HAS_LANGDETECT = False
    detect = None

HAS_DEEP_TRANSLATOR = True
try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
except Exception:
    HAS_DEEP_TRANSLATOR = False
    DeepGoogleTranslator = None

HAS_GOOGLETRANS = True
try:
    from googletrans import Translator as GoogleTranslator
except Exception:
    HAS_GOOGLETRANS = False
    GoogleTranslator = None

HAS_SPACY = True
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    HAS_SPACY = False
    nlp = None

HAS_SPEECH = True
try:
    import speech_recognition as sr
    import pyttsx3
except Exception:
    HAS_SPEECH = False

# ------------------------
# Config / filenames
# ------------------------
APP_TITLE = "🎓 Advanced Deep NLP Chatbot"
LOG_FILE = "chatbot_logs.csv"
HISTORY_FILE = "chat_history.csv"
FAQ_FILE = "faq.csv"
INTENTS_FILE = "intents.json"
DATA_PTH = "data.pth"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CONTEXT = 5
MAX_SENT_LEN = 16
SIM_THRESHOLD = 0.62
PROB_THRESHOLD = 0.70

# ------------------------
# Custom CSS for styling
# ------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Chat containers */
    .user-message {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        position: relative;
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
        position: relative;
    }
    
    /* Message metadata */
    .message-meta {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 4px;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 20px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6e8efb;
        color: white;
    }
    
    /* Metrics */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 4px solid #6e8efb;
    }
    
    /* Voice button */
    .voice-btn {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .voice-btn:hover {
        transform: scale(1.1);
    }
    
    /* Feedback buttons */
    .feedback-btn {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0 5px;
        transition: all 0.3s ease;
    }
    
    .feedback-btn:hover {
        transform: scale(1.2);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #4CAF50;
    }
    
    .status-offline {
        background-color: #f44336;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------
# Safe CSV init
# ------------------------
def ensure_csv(path, header):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(header)
    except Exception as e:
        print(f"[ensure_csv] Failed to create {path}: {e}")

ensure_csv(LOG_FILE, ["timestamp", "user_input", "user_lang", "translated_input", "predicted_tag", "response", "feedback", "confidence", "detected_lang", "translated_from"])
ensure_csv(HISTORY_FILE, ["timestamp", "speaker", "message"])
ensure_csv("ratings.csv", ["timestamp", "rating"])

# ------------------------
# Load intents & optional FAQ
# ------------------------
if os.path.exists(INTENTS_FILE):
    try:
        with open(INTENTS_FILE, "r", encoding="utf-8") as f:
            intents = json.load(f)
    except Exception:
        intents = {"intents": []}
else:
    intents = {"intents": []}

faq_df = None
if os.path.exists(FAQ_FILE):
    try:
        faq_df = pd.read_csv(FAQ_FILE)
    except Exception:
        faq_df = None

# ------------------------
# Embeddings (SBERT) - cached
# ------------------------
@st.cache_resource
def load_embedder(model_name=EMBED_MODEL_NAME):
    if not HAS_SBERT:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None

embedder = load_embedder() if HAS_SBERT else None

# Precompute intent-pattern embeddings
intent_pattern_embeddings = []
if embedder and intents.get("intents"):
    for intent in intents["intents"]:
        patterns = intent.get("patterns", [])
        emb = None
        try:
            if patterns:
                emb = embedder.encode(patterns, convert_to_tensor=True)
        except Exception:
            emb = None
        intent_pattern_embeddings.append({"tag": intent.get("tag"), "emb": emb, "responses": intent.get("responses", [])})
else:
    for intent in intents.get("intents", []):
        intent_pattern_embeddings.append({"tag": intent.get("tag"), "emb": None, "responses": intent.get("responses", [])})

# Precompute FAQ embeddings if available
faq_embeddings = None
if embedder and faq_df is not None and not faq_df.empty:
    try:
        faq_embeddings = embedder.encode(faq_df['question'].astype(str).tolist(), convert_to_tensor=True)
    except Exception:
        faq_embeddings = None

# ------------------------
# Translation / detection helpers
# ------------------------
def detect_language_safe(text: str) -> str:
    if not text or not text.strip():
        return "en"
    if not HAS_LANGDETECT:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_to_en(text: str, src: str = None):
    detected_lang = src or detect_language_safe(text)
    if detected_lang == "en":
        return text, "en"
    
    # Try deep_translator first
    if HAS_DEEP_TRANSLATOR:
        try:
            # Map language codes if needed
            lang_map = {"zh-cn": "zh-CN", "pcm": "en"}  # Nigerian Pidgin not supported, default to English
            source_lang = lang_map.get(detected_lang, detected_lang)
            
            if source_lang != "en":
                tr = DeepGoogleTranslator(source=source_lang, target="en")
                return tr.translate(text), detected_lang
        except Exception as e:
            st.sidebar.warning(f"Deep translation failed: {str(e)}")
    
    # Fallback to googletrans
    if HAS_GOOGLETRANS:
        try:
            tr = GoogleTranslator()
            res = tr.translate(text, src=detected_lang, dest="en")
            if hasattr(res, "text"):
                return res.text, detected_lang
            else:
                return str(res), detected_lang
        except Exception as e:
            st.sidebar.warning(f"Google translation failed: {str(e)}")
    
    return text, detected_lang

def translate_from_en(text: str, target: str) -> str:
    if not text or not text.strip():
        return text
    if not target or target == "en":
        return text
    
    # Map language codes if needed
    lang_map = {"zh-cn": "zh-CN", "pcm": "en"}  # Nigerian Pidgin not supported, default to English
    target_lang = lang_map.get(target, target)
    
    if target_lang == "en":
        return text
    
    # Try deep_translator first
    if HAS_DEEP_TRANSLATOR:
        try:
            tr = DeepGoogleTranslator(source="en", target=target_lang)
            return tr.translate(text)
        except Exception as e:
            st.sidebar.warning(f"Deep translation failed: {str(e)}")
    
    # Fallback to googletrans
    if HAS_GOOGLETRANS:
        try:
            tr = GoogleTranslator()
            res = tr.translate(text, src="en", dest=target_lang)
            if hasattr(res, "text"):
                return res.text
            else:
                return str(res)
        except Exception as e:
            st.sidebar.warning(f"Google translation failed: {str(e)}")
    
    return text

# ------------------------
# Text cleaning / lemmatization
# ------------------------
def clean_text(text):
    if text is None:
        return ""
    s = text.strip()
    s = s.lower()
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def lemmatize_text(text):
    if not HAS_SPACY or nlp is None:
        return text
    try:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    except Exception:
        return text

# ------------------------
# Semantic matchers
# ------------------------
def semantic_intent_match(text):
    if embedder is None:
        return None, 0.0, None
    try:
        u_emb = embedder.encode(text, convert_to_tensor=True)
    except Exception:
        return None, 0.0, None
    best_tag, best_score, best_resp = None, 0.0, None
    for item in intent_pattern_embeddings:
        if item["emb"] is None:
            continue
        scores = util.cos_sim(u_emb, item["emb"])[0]
        value = float(scores.max())
        if value > best_score:
            best_score = value
            best_tag = item["tag"]
            if item["responses"]:
                best_resp = random.choice(item["responses"])
    return best_tag, best_score, best_resp

def semantic_faq_match(text):
    if faq_embeddings is None or embedder is None or faq_df is None:
        if faq_df is not None:
            for _, row in faq_df.iterrows():
                q = str(row.get('question', ''))
                if q and q.lower() in text.lower():
                    return str(row.get('answer', '')), 1.0
        return None, 0.0
    try:
        u_emb = embedder.encode(text, convert_to_tensor=True)
        sims = util.cos_sim(u_emb, faq_embeddings)[0]
        idx = int(np.argmax(sims))
        sc = float(sims[idx])
        if sc >= SIM_THRESHOLD:
            return str(faq_df.iloc[idx]['answer']), sc
        return None, sc
    except Exception:
        return None, 0.0

def keyword_intent_match(text):
    t = clean_text(text)
    for intent in intents.get("intents", []):
        for p in intent.get("patterns", []):
            if p and p.lower() in t:
                return intent.get("tag"), 0.5, random.choice(intent.get("responses", ["I can help with that."]))
    return None, 0.0, None

# ------------------------
# Optional PyTorch model loading
# ------------------------
model = None
word2idx = {}
tags = []
if HAS_TORCH and os.path.exists(DATA_PTH):
    try:
        data = torch.load(DATA_PTH, map_location=torch.device("cpu"))
        word2idx = data.get("word2idx", {})
        tags = data.get("tags", [])
        try:
            from model import DeepChatbot
            model = DeepChatbot(data["vocab_size"], data["embed_dim"], data["hidden_size"], len(tags))
            model.load_state_dict(data["model_state"])
            model.eval()
        except Exception:
            model = None
    except Exception:
        model = None

def model_predict_intent(text):
    if model is None or not word2idx:
        return None, 0.0
    tokens = clean_text(text).split()
    ids = [word2idx.get(tok, word2idx.get("<UNK>", 0)) for tok in tokens][:MAX_SENT_LEN]
    if len(ids) < MAX_SENT_LEN:
        ids += [0] * (MAX_SENT_LEN - len(ids))
    X = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1)
        p, idx = torch.max(probs, dim=1)
        tag = tags[idx.item()] if tags else None
        return tag, float(p.item())
    return None, 0.0

# ------------------------
# NER (spaCy)
# ------------------------
def extract_entities(text):
    if not HAS_SPACY or nlp is None:
        return []
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception:
        return []

# ------------------------
# Special commands
# ------------------------
def special_commands(msg):
    if not msg:
        return None
    if msg.startswith("/book"):
        parts = msg.split(maxsplit=1)
        item = parts[1] if len(parts) > 1 else "General Service"
        return ("booking", f"✅ Booking confirmed for {item}. We will contact you.")
    if msg.startswith("/recommend"):
        return ("recommendation", "📌 Recommendation: Premium plan + warranty.")
    if msg.startswith("/troubleshoot"):
        return ("troubleshoot", "🛠️ Try restarting the device; if issue persists, contact support.")
    return None

# ------------------------
# Speech functions
# ------------------------
def speak_text(text):
    if not HAS_SPEECH:
        return False
    
    try:
        engine = pyttsx3.init()
        
        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed percent
        engine.setProperty('volume', 0.9)  # Volume 0-1
        
        # Try to set a more natural voice if available
        voices = engine.getProperty('voices')
        if voices:
            # Prefer female voice if available
            for voice in voices:
                if "female" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        st.sidebar.error(f"Text-to-speech error: {str(e)}")
        return False

def recognize_speech():
    if not HAS_SPEECH:
        return None, "Speech recognition not available"
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... (speak now)")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=8)
        
        text = r.recognize_google(audio)
        return text, None
    except sr.WaitTimeoutError:
        return None, "No speech detected within timeout"
    except sr.UnknownValueError:
        return None, "Could not understand audio"
    except sr.RequestError as e:
        return None, f"Recognition error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# ------------------------
# Logging helpers
# ------------------------
def log_interaction(user_input, user_lang, translated_input, predicted_tag, response, feedback=None, confidence=None, detected_lang=None, translated_from=None):
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        user_input,
                        user_lang,
                        translated_input,
                        predicted_tag,
                        response,
                        feedback,
                        confidence,
                        detected_lang,
                        translated_from])
    except Exception:
        pass

def log_history(speaker, message):
    try:
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speaker, message])
    except Exception:
        pass

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
inject_custom_css()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.title("🤖 Smart Chatbot (NLP)")
st.sidebar.info("Try /book, /recommend, /troubleshoot. Use the tabs for Evaluation/History/Settings.")

# --- Sidebar: Translation selector ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Bot Response Language")
language_options = {
    "English 🇬🇧": "en",
    "Chinese 🇨🇳": "zh-cn",
    "German 🇩🇪": "de",
    "French 🇫🇷": "fr",
    "Hindi 🇮🇳": "hi",
    "Spanish 🇪🇸": "es",
    "Portuguese 🇵🇹": "pt",
    "Russian 🇷🇺": "ru",
    "Nigerian Pidgin 🇳🇬": "pcm"
}
selected_lang_display = st.sidebar.selectbox("Select target language for bot responses:", list(language_options.keys()))
TARGET_LANG_CODE = language_options[selected_lang_display]

# Status indicators
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Status")
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if embedder else 'status-offline'}'></span> **SBERT Embeddings:** {'Available' if embedder else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if (HAS_DEEP_TRANSLATOR or HAS_GOOGLETRANS) else 'status-offline'}'></span> **Translation:** {'Available' if (HAS_DEEP_TRANSLATOR or HAS_GOOGLETRANS) else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if HAS_LANGDETECT else 'status-offline'}'></span> **Language Detection:** {'Available' if HAS_LANGDETECT else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if model else 'status-offline'}'></span> **PyTorch Model:** {'Loaded' if model else 'Not Loaded'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if HAS_SPEECH else 'status-offline'}'></span> **Speech I/O:** {'Available' if HAS_SPEECH else 'Not Available'}", unsafe_allow_html=True)

st.title(APP_TITLE)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chatbot", "📊 Evaluation", "📜 Chat History", "⚙️ Settings / Rating"])

# session init
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = deque(maxlen=MAX_CONTEXT)
if "speak_replies" not in st.session_state:
    st.session_state["speak_replies"] = False
if "listening" not in st.session_state:
    st.session_state["listening"] = False

# --- Chatbot Tab ---
with tab1:
    st.subheader("💬 Chat")
    
    # Input area with columns
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.chat_input("Type your message here...") if hasattr(st, "chat_input") else st.text_input("Type your message here...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if HAS_SPEECH:
            if st.button("🎤 Use Microphone", key="mic_btn", use_container_width=True):
                st.session_state["listening"] = True
        
        # Additional options
        st.session_state["speak_replies"] = st.checkbox("🔊 Speak replies", value=st.session_state["speak_replies"])
    
    # Handle speech recognition
    if st.session_state.get("listening", False):
        recognized_text, error = recognize_speech()
        if recognized_text:
            user_input = recognized_text
            st.success(f"Recognized: {recognized_text}")
        elif error:
            st.error(f"Recognition error: {error}")
        st.session_state["listening"] = False

    if user_input:
        user_lang = detect_language_safe(user_input) if HAS_LANGDETECT else "en"
        translated_input, translated_from = translate_to_en(user_input, src=user_lang)

        proc_text = lemmatize_text(clean_text(translated_input))

        tag = None
        response = None
        conf = 0.0

        sc = special_commands(user_input)
        if sc:
            tag, response = sc
            conf = 1.0
        else:
            faq_ans, faq_score = semantic_faq_match(proc_text)
            if faq_ans and faq_score >= SIM_THRESHOLD:
                tag = "faq"
                response = faq_ans
                conf = faq_score
            else:
                if model is not None:
                    try:
                        m_tag, m_conf = model_predict_intent(proc_text)
                        if m_tag is not None and m_conf >= PROB_THRESHOLD:
                            tag = m_tag
                            # Find the intent and get a response from it
                            for it in intents.get("intents", []):
                                if it.get("tag") == tag:
                                    response = random.choice(it.get("responses", ["I can help with that."]))
                                    break
                            conf = m_conf
                    except Exception:
                        pass

                if tag is None:
                    s_tag, s_score, s_resp = semantic_intent_match(proc_text)
                    if s_tag and s_score >= SIM_THRESHOLD:
                        tag = s_tag
                        response = s_resp if s_resp else (random.choice([r for it in intents.get("intents", []) if it.get("tag") == s_tag for r in it.get("responses", [])]) if intents.get("intents") else "I can help.")
                        conf = s_score

                if tag is None:
                    k_tag, k_score, k_resp = keyword_intent_match(proc_text)
                    if k_tag:
                        tag = k_tag
                        response = k_resp
                        conf = k_score

                if tag is None:
                    tag = "unknown"
                    # Provide more helpful unknown responses based on context
                    unknown_responses = [
                        "🤔 I'm not sure I understand. Could you rephrase that?",
                        "🔍 I'm still learning. Could you try asking in a different way?",
                        "❓ I didn't catch that. Can you provide more details?",
                        "💡 That's an interesting question. Let me check my knowledge base and get back to you."
                    ]
                    response = random.choice(unknown_responses)
                    conf = 0.0

        entities = extract_entities(proc_text)
        if tag == "booking" and "{item}" in str(response):
            # Extract the item from the user input if possible
            item_match = re.search(r"/book\s+(.+)", user_input, re.IGNORECASE)
            item = item_match.group(1) if item_match else "your selected service"
            response = str(response).replace("{item}", item)

        final_response = translate_from_en(response, TARGET_LANG_CODE) if TARGET_LANG_CODE != "en" else response

        st.session_state["messages"].append(("You", user_input, None, None, user_lang))
        st.session_state["messages"].append(("Bot", final_response, tag, conf, selected_lang_display))
        st.session_state["context"].append(user_input)
        log_history("User", user_input)
        log_history("Bot", final_response)
        log_interaction(user_input, user_lang, translated_input, tag, final_response, None, conf, user_lang, translated_from)

        if st.session_state["speak_replies"]:
            speak_success = speak_text(final_response)
            if not speak_success:
                st.sidebar.warning("Text-to-speech failed. Please check your audio settings.")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, (speaker, text, tag, conf, lang) in enumerate(st.session_state["messages"]):
            if speaker == "You":
                st.markdown(f"""
                <div class="user-message">
                    🧑 <b>You</b>: {text}
                    <div class="message-meta">Language: {lang} • {datetime.now().strftime("%H:%M:%S")}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    🤖 <b>Bot</b>: {text}
                    <div class="message-meta">Intent: {tag if tag else 'N/A'} • Confidence: {conf:.2%} • Language: {lang}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback buttons for the last message only
                if i == len(st.session_state["messages"]) - 1:
                    col_a, col_b, col_c = st.columns([1, 6, 2])
                    with col_a:
                        if st.button("👍", key=f"yes_{i}", help="Response was helpful"):
                            prev_user = None
                            for j in range(i - 1, -1, -1):
                                if st.session_state["messages"][j][0] == "You":
                                    prev_user = st.session_state["messages"][j][1]
                                    break
                            if prev_user:
                                log_interaction(prev_user, st.session_state["messages"][j][4], None, 
                                              st.session_state["messages"][i][2], text, "yes", conf, lang, None)
                                st.success("Thanks for the feedback!")
                    with col_b:
                        if st.button("👎", key=f"no_{i}", help="Response was not helpful"):
                            prev_user = None
                            for j in range(i - 1, -1, -1):
                                if st.session_state["messages"][j][0] == "You":
                                    prev_user = st.session_state["messages"][j][1]
                                    break
                            if prev_user:
                                log_interaction(prev_user, st.session_state["messages"][j][4], None, 
                                              st.session_state["messages"][i][2], text, "no", conf, lang, None)
                                st.error("Feedback saved. We'll improve!")
                    with col_c:
                        if st.session_state["speak_replies"] and st.button("🔊", key=f"speak_{i}", help="Repeat this response"):
                            speak_text(text)

# --- Evaluation Tab ---
with tab2:
    st.subheader("📊 Evaluation & Analytics")
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
        except Exception as e:
            st.warning(f"Could not read log file: {e}")
            df = pd.DataFrame()
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Interactions", len(df))
            if "feedback" in df.columns:
                df_fb = df[df["feedback"].notna()]
                if not df_fb.empty:
                    pos = df_fb["feedback"].astype(str).str.lower().isin(["yes","1","y","true"]).sum()
                    tot = len(df_fb)
                    with col2:
                        st.metric("Feedback Samples", tot)
                    with col3:
                        st.metric("Positive Feedback", f"{pos} ({pos/tot:.1%})")
            
            st.subheader("Interactions by Intent")
            if "predicted_tag" in df.columns:
                summary = df.groupby("predicted_tag").size().reset_index(name="count").sort_values("count", ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(summary)))
                ax.bar(summary["predicted_tag"], summary["count"], color=colors)
                ax.set_xticklabels(summary["predicted_tag"], rotation=45, ha="right")
                ax.set_ylabel("Count")
                ax.set_title("Interactions by Intent")
                st.pyplot(fig)
            
            st.subheader("Daily Interaction Trends")
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                ts = df.set_index("timestamp").resample("D").size()
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(ts.index, ts.values, marker="o", color='orange', linewidth=2, markersize=4)
                ax2.fill_between(ts.index, ts.values, alpha=0.3, color='orange')
                ax2.set_title("Daily Interactions")
                ax2.set_ylabel("Count")
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            except Exception:
                pass
            
            st.subheader("Confidence Distribution")
            if "confidence" in df.columns:
                try:
                    conf_df = df[df["confidence"].notna()]
                    if not conf_df.empty:
                        fig3, ax3 = plt.subplots(figsize=(10, 4))
                        ax3.hist(conf_df["confidence"].astype(float), bins=20, alpha=0.7, color='purple', edgecolor='black')
                        ax3.set_xlabel("Confidence")
                        ax3.set_ylabel("Frequency")
                        ax3.set_title("Confidence Distribution of Responses")
                        st.pyplot(fig3)
                except Exception:
                    pass
            
            col_a, col_b = st.columns(2)
            with col_a:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Evaluation Logs", csv_bytes, "chatbot_logs.csv", "text/csv")
            if os.path.exists("ratings.csv"):
                with col_b:
                    ratings_df = pd.read_csv("ratings.csv", on_bad_lines="skip")
                    st.download_button("📥 Download Ratings", ratings_df.to_csv(index=False).encode("utf-8"), "ratings.csv", "text/csv")
        else:
            st.info("No logs yet.")
    else:
        st.info("Log file not found.")

# --- Chat History Tab ---
with tab3:
    st.subheader("📜 Conversation History")
    df = pd.read_csv(HISTORY_FILE, on_bad_lines="skip") if os.path.exists(HISTORY_FILE) else pd.DataFrame()
    if not df.empty:
        # Display in a more chat-like format
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row['timestamp']).strftime("%H:%M:%S") if 'timestamp' in row else "N/A"
            if row['speaker'] == 'User':
                st.markdown(f"""
                <div class="user-message">
                    🧑 <b>User</b>: {row["message"]}
                    <div class="message-meta">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    🤖 <b>Bot</b>: {row["message"]}
                    <div class="message-meta">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            csv_history = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Chat History", csv_history, "chat_history.csv", "text/csv", key="download-chat-history")
        with col2:
            if os.path.exists("ratings.csv"):
                ratings_df = pd.read_csv("ratings.csv", on_bad_lines="skip")
                st.download_button("📥 Download Ratings", ratings_df.to_csv(index=False).encode("utf-8"), "ratings.csv", "text/csv", key="download-ratings")
    else:
        st.info("No chat history yet.")

# --- Settings / Rating Tab ---
with tab4:
    st.subheader("⚙️ Settings & Rating")
    
    st.info("Configure the chatbot behavior and provide feedback on your experience.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**System Configuration**")
        st.write(f"Sentence-BERT available: {'✅' if bool(embedder) else '❌'}")
        st.write(f"spaCy loaded: {'✅' if bool(nlp) else '❌'}")
        st.write(f"Language detect available: {'✅' if HAS_LANGDETECT else '❌'}")
        st.write(f"GoogleTrans available: {'✅' if HAS_GOOGLETRANS else '❌'}")
        st.write(f"DeepTranslator available: {'✅' if HAS_DEEP_TRANSLATOR else '❌'}")
        st.write(f"Voice I/O: {'✅' if HAS_SPEECH else '❌'}")
        
        st.divider()
        
        st.write("**Performance Settings**")
        sim_val = st.slider("Semantic similarity threshold", 0.4, 0.9, float(SIM_THRESHOLD), 0.01)
        prob_val = st.slider("Probability threshold", 0.5, 0.95, float(PROB_THRESHOLD), 0.01)
        
        if st.button("Apply Settings"):
            SIM_THRESHOLD = sim_val
            PROB_THRESHOLD = prob_val
            st.success(f"Applied similarity threshold = {SIM_THRESHOLD:.2f}, probability threshold = {PROB_THRESHOLD:.2f}")
    
    with col2:
        st.write("**Rate Your Experience**")
        st.write("How would you rate your experience with our chatbot?")
        
        rating = st.radio(
            "Select a rating:",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            horizontal=True
        )
        
        feedback_text = st.text_area("Additional feedback (optional):")
        
        if st.button("Submit Rating"):
            ensure_csv("ratings.csv", ["timestamp", "rating"])
            with open("ratings.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating])
            
            if feedback_text:
                with open("feedback.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Rating: {rating}/5\nFeedback: {feedback_text}\n")
            
            st.success(f"Thanks for your feedback! You rated us {rating} ⭐")
            
            if rating <= 2:
                st.info("We're sorry to hear about your experience. Our team will review your feedback.")

st.markdown("---")
st.caption("Built with semantic embeddings + optional PyTorch model. Logs: chatbot_logs.csv, chat_history.csv.")
