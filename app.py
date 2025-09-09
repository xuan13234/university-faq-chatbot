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
    from langdetect import detect
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
# Safe CSV init
# ------------------------
def ensure_csv(path, header):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(header)

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
# Embeddings (SBERT)
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
    except Exception as e:
        st.warning(f"[Language detection failed] {e}")
        return "en"

def translate_to_en(text: str, src: str = None):
    detected_lang = src or detect_language_safe(text)
    if detected_lang == "en":
        return text, "en"
    if HAS_DEEP_TRANSLATOR:
        try:
            tr = DeepGoogleTranslator(source=detected_lang, target="en")
            return tr.translate(text), detected_lang
        except Exception as e:
            st.warning(f"[DeepTranslator failed] {e}")
    if HAS_GOOGLETRANS:
        try:
            tr = GoogleTranslator()
            res = tr.translate(text, src=detected_lang, dest="en")
            return getattr(res, "text", res), detected_lang
        except Exception as e:
            st.warning(f"[GoogleTrans failed] {e}")
    st.info("Translation not available; using original text.")
    return text, detected_lang

def translate_from_en(text: str, target: str) -> str:
    if not text or not text.strip() or target == "en":
        return text
    if HAS_DEEP_TRANSLATOR:
        try:
            tr = DeepGoogleTranslator(source="en", target=target)
            return tr.translate(text)
        except Exception as e:
            st.warning(f"[DeepTranslator (from_en) failed] {e}")
    if HAS_GOOGLETRANS:
        try:
            tr = GoogleTranslator()
            res = tr.translate(text, src="en", dest=target)
            return getattr(res, "text", res)
        except Exception as e:
            st.warning(f"[GoogleTrans (from_en) failed] {e}")
    st.info("Translation not available; using original text.")
    return text

# ------------------------
# Text cleaning / lemmatization
# ------------------------
def clean_text(text):
    if text is None:
        return ""
    s = text.strip().lower()
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
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.title("🤖 Smart Chatbot (NLP)")
st.sidebar.info("Try /book, /recommend, /troubleshoot. Use the tabs for Evaluation/History/Settings.")

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

st.title(APP_TITLE)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chatbot", "📊 Evaluation", "📜 Chat History", "⚙️ Settings / Rating"])

# session init
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = deque(maxlen=MAX_CONTEXT)
