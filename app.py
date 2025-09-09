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
# Optional heavy libraries (import-if-available)
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

# Prefer deep_translator (reliable) and fall back to googletrans if available
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
LOG_FILE = "chatbot_logs.csv"            # logs with feedback
HISTORY_FILE = "chat_history.csv"       # chronological chat history
FAQ_FILE = "faq.csv"                    # optional: question,answer
INTENTS_FILE = "intents.json"           # intents definitions
DATA_PTH = "data.pth"                   # optional PyTorch artifact with model_state etc.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CONTEXT = 5
MAX_SENT_LEN = 16
SIM_THRESHOLD = 0.62
PROB_THRESHOLD = 0.70

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

# Precompute intent-pattern embeddings when possible
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
# Translation / detection helpers (improved)
# ------------------------
def detect_language_safe(text: str) -> str:
    """Detect language safely, default to 'en' if unknown."""
    if not text or not text.strip():
        return "en"
    if not HAS_LANGDETECT:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_to_en(text: str, src: str = None):
    """
    Translate text to English.
    Returns (translated_text, detected_source_language).
    """
    if not text or not text.strip():
        return text, "en"

    detected_lang = src or detect_language_safe(text)
    if detected_lang == "en":
        return text, "en"

    if HAS_DEEP_TRANSLATOR:
        try:
            tr = DeepGoogleTranslator(source=detected_lang, target="en")
            return tr.translate(text), detected_lang
        except Exception:
            pass

    if HAS_GOOGLETRANS:
        try:
            tr = GoogleTranslator()
            # googletrans API may vary; in older versions .translate returns object with .text
            res = tr.translate(text, src=detected_lang, dest="en")
            return getattr(res, "text", res), detected_lang
        except Exception:
            pass

    return text, detected_lang

def translate_from_en(text: str, target: str) -> str:
    """Translate from English to target language."""
    if not text or not text.strip():
        return text
    if not target or target == "en":
        return text

    if HAS_DEEP_TRANSLATOR:
        try:
            tr = DeepGoogleTranslator(source="en", target=target)
            return tr.translate(text)
        except Exception:
            pass

    if HAS_GOOGLETRANS:
        try:
            tr = GoogleTranslator()
            res = tr.translate(text, src="en", dest=target)
            return getattr(res, "text", res)
        except Exception:
            pass

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
    """Return (tag, score, response) using SBERT if available"""
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
    """Return (answer, score) if near an FAQ"""
    if faq_embeddings is None or embedder is None or faq_df is None:
        # fallback substring
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

# ------------------------
# Keyword fallback
# ------------------------
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
            # attempt to import user's DeepChatbot class (model.py)
            from model import DeepChatbot  # type: ignore
            model = DeepChatbot(data["vocab_size"], data["embed_dim"], data["hidden_size"], len(tags))
            model.load_state_dict(data["model_state"])
            model.eval()
        except Exception:
            model = None
    except Exception:
        model = None

def model_predict_intent(text):
    """Predict using user-provided PyTorch model (if loaded)"""
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
# Logging helpers (single, consistent version)
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
# Streamlit UI (tabs)
# ------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.title("🤖 Smart Chatbot (NLP)")
st.sidebar.info("Try /book, /recommend, /troubleshoot. Use the tabs for Evaluation/History/Settings.")

st.title(APP_TITLE)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chatbot", "📊 Evaluation", "📜 Chat History", "⚙️ Settings / Rating"])

# session initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []   # (speaker, text, tag, conf, lang)
if "context" not in st.session_state:
    st.session_state["context"] = deque(maxlen=MAX_CONTEXT)

# --- Chatbot Tab ---
with tab1:
    st.subheader("💬 Chat")
    user_input = st.chat_input("Type your message here...") if hasattr(st, "chat_input") else st.text_input("Type your message here...")
    cols = st.columns([1, 3])
    with cols[0]:
        if HAS_SPEECH:
            if st.button("🎤 Speak (microphone)"):
                try:
                    r = sr.Recognizer()
                    with sr.Microphone() as source:
                        st.info("🎙️ Listening...")
                        audio = r.listen(source, timeout=5, phrase_time_limit=8)
                    spoken = r.recognize_google(audio)
                    user_input = spoken
                    st.success(f"You said: {spoken}")
                except Exception as e:
                    st.error(f"Voice capture failed: {e}")
        else:
            st.caption("Voice I/O not available")
    with cols[1]:
        speak_replies = st.checkbox("🔊 Speak replies", value=False)

    # single, consistent user_input handling block
    if user_input:
        # detect language and translate to en (working language)
        user_lang = detect_language_safe(user_input) if HAS_LANGDETECT else "en"
        translated_input, translated_from = translate_to_en(user_input, src=user_lang)

        proc_text = lemmatize_text(clean_text(translated_input))

        # decide response & tag & confidence
        tag = None
        response = None
        conf = 0.0

        # special commands -> immediate response (use original user_input)
        sc = special_commands(user_input)
        if sc:
            tag, response = sc
            conf = 1.0
        else:
            # 1) semantic FAQ
            faq_ans, faq_score = semantic_faq_match(proc_text)
            if faq_ans and faq_score >= SIM_THRESHOLD:
                tag = "faq"
                response = faq_ans
                conf = faq_score
            else:
                # 2) supervised model (if available)
                if model is not None:
                    try:
                        m_tag, m_conf = model_predict_intent(proc_text)
                        if m_tag is not None and m_conf >= PROB_THRESHOLD:
                            tag = m_tag
                            # choose response from intents if available
                            for it in intents.get("intents", []):
                                if it.get("tag") == tag:
                                    response = random.choice(it.get("responses", ["I can help with that."]))
                                    break
                            conf = m_conf
                    except Exception:
                        pass

                # 3) semantic intent match
                if tag is None:
                    s_tag, s_score, s_resp = semantic_intent_match(proc_text)
                    if s_tag and s_score >= SIM_THRESHOLD:
                        tag = s_tag
                        response = s_resp if s_resp else (random.choice([r for it in intents.get("intents", []) if it.get("tag") == s_tag for r in it.get("responses", [])]) if intents.get("intents") else "I can help.")
                        conf = s_score

                # 4) keyword fallback
                if tag is None:
                    k_tag, k_score, k_resp = keyword_intent_match(proc_text)
                    if k_tag:
                        tag = k_tag
                        response = k_resp
                        conf = k_score

                # 5) ultimate fallback
                if tag is None:
                    tag = "unknown"
                    response = "🤔 Sorry, I didn't quite understand. Could you rephrase?"
                    conf = 0.0

        # entities (optional)
        entities = extract_entities(proc_text)
        if tag == "booking" and "{item}" in str(response):
            response = str(response).replace("{item}", "your selected service")

        # translate response back to user's language if needed
        final_response = translate_from_en(response, user_lang) if (user_lang and user_lang != "en") else response

        # update session & logs
        st.session_state["messages"].append(("You", user_input, None, None, user_lang))
        st.session_state["messages"].append(("Bot", final_response, tag, conf, user_lang))
        st.session_state["context"].append(user_input)
        log_history("User", user_input)
        log_history("Bot", final_response)
        log_interaction(user_input, user_lang, translated_input, tag, final_response, None, conf, user_lang, translated_from)

        # speak replies
        if speak_replies and HAS_SPEECH:
            try:
                tts = pyttsx3.init()
                tts.say(final_response)
                tts.runAndWait()
            except Exception:
                pass

    # render messages (fixed)
    for i, (speaker, text, tag, conf, lang) in enumerate(st.session_state["messages"]):
        if speaker == "You":
            st.chat_message("user").markdown(
                f"<div style='background:#e6f0ff;padding:8px;border-radius:10px;'>🧑 {text} <small>({lang})</small></div>",
                unsafe_allow_html=True
            )
        else:
            st.chat_message("assistant").markdown(
                f"<div style='background:#f2f2f2;padding:8px;border-radius:10px;'>🤖 {text} <small>({lang})</small></div>",
                unsafe_allow_html=True
            )
            # feedback buttons for latest bot response
            if i == len(st.session_state["messages"]) - 1:
                c1, c2 = st.columns([1,1])
                with c1:
                    if st.button("👍 Correct", key=f"yes_{i}"):
                        prev_user = None
                        for j in range(i - 1, -1, -1):
                            if st.session_state["messages"][j][0] == "You":
                                prev_user = st.session_state["messages"][j][1]
                                break
                        log_interaction(prev_user, st.session_state["messages"][j][4], None, st.session_state["messages"][i][2], text, "yes", conf, lang, None)
                        st.success("Thanks for the feedback!")
                with c2:
                    if st.button("👎 Incorrect", key=f"no_{i}"):
                        prev_user = None
                        for j in range(i - 1, -1, -1):
                            if st.session_state["messages"][j][0] == "You":
                                prev_user = st.session_state["messages"][j][1]
                                break
                        log_interaction(prev_user, st.session_state["messages"][j][4], None, st.session_state["messages"][i][2], text, "no", conf, lang, None)
                        st.error("Feedback saved.")

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
            st.metric("Total logged interactions", len(df))
            if "feedback" in df.columns:
                df_fb = df[df["feedback"].notna()]
                if not df_fb.empty:
                    pos = df_fb["feedback"].astype(str).str.lower().isin(["yes","1","y","true"]).sum()
                    tot = len(df_fb)
                    st.metric("Feedback samples", tot)
                    st.metric("Positive feedback", f"{pos} ({pos/tot:.2%})")
            if "predicted_tag" in df.columns:
                summary = df.groupby("predicted_tag").size().reset_index(name="count").sort_values("count", ascending=False)
                st.write("### Interactions by intent")
                st.dataframe(summary)
                # plot
                fig, ax = plt.subplots(figsize=(7,4))
                ax.bar(summary["predicted_tag"], summary["count"])
                ax.set_xticklabels(summary["predicted_tag"], rotation=30, ha="right")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            # timeseries
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                ts = df.set_index("timestamp").resample("D").size()
                fig2, ax2 = plt.subplots(figsize=(8,3))
                ax2.plot(ts.index, ts.values, marker="o")
                ax2.set_title("Daily interactions")
                ax2.set_ylabel("Count")
                st.pyplot(fig2)
            except Exception:
                pass
            # download evaluation logs
            col_a, col_b = st.columns(2)
            with col_a:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Evaluation Logs", csv_bytes, "chatbot_logs.csv", "text/csv")
            # show ratings if exists
            if os.path.exists("ratings.csv"):
                with col_b:
                    ratings_df = pd.read_csv("ratings.csv", on_bad_lines="skip")
                    st.download_button("⬇️ Download Ratings", ratings_df.to_csv(index=False).encode("utf-8"), "ratings.csv", "text/csv")
        else:
            st.info("No logs yet.")
    else:
        st.info("Log file not found.")

# --- Chat History Tab ---
with tab3:
    st.subheader("📜 Conversation History")
    df = pd.read_csv(HISTORY_FILE, on_bad_lines="skip") if os.path.exists(HISTORY_FILE) else pd.DataFrame()
    if not df.empty:
        st.dataframe(df)
        col1, col2 = st.columns(2)
        with col1:
            csv_history = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Chat History", csv_history, "chat_history.csv", "text/csv", key="download-chat-history")
        with col2:
            if os.path.exists("ratings.csv"):
                ratings_df = pd.read_csv("ratings.csv", on_bad_lines="skip")
                st.download_button("⬇️ Download Ratings", ratings_df.to_csv(index=False).encode("utf-8"), "ratings.csv", "text/csv", key="download-ratings")
    else:
        st.info("No chat history yet.")

# --- Settings / Rating Tab ---
with tab4:
    st.subheader("⚙️ Settings & Rating")
    st.write("**Available features**")
    st.write(f"Sentence-BERT available: {bool(embedder)}")
    st.write(f"spaCy loaded: {bool(nlp)}")
    st.write(f"Language detect available: {HAS_LANGDETECT}")
    st.write(f"GoogleTrans available: {HAS_GOOGLETRANS}")
    st.write(f"DeepTranslator available: {HAS_DEEP_TRANSLATOR}")
    st.write(f"Optional voice I/O: {HAS_SPEECH}")
    sim_val = st.slider("Semantic similarity threshold", 0.4, 0.9, float(SIM_THRESHOLD), 0.01)
    if st.button("Apply thresholds"):
        SIM_THRESHOLD = sim_val
        st.success(f"Applied similarity threshold = {SIM_THRESHOLD:.2f}")
    st.markdown("---")
    rating = st.slider("Rate the chatbot", 1, 5, 4)
    if st.button("Submit rating"):
        ensure_csv("ratings.csv", ["timestamp", "rating"])
        with open("ratings.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating])
        st.success(f"Thanks! You rated {rating} ⭐")

st.markdown("---")
st.caption("Built with semantic embeddings + optional PyTorch model. Logs: chatbot_logs.csv, chat_history.csv.")

