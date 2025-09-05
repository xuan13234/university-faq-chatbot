import streamlit as st
import joblib, random, json, re
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect
import datetime
import plotly.express as px
import pandas as pd

# =============================
# Configuration & Paths
# =============================
DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
LOG_PATH = Path(__file__).resolve().parent / "data" / "chat_logs.json"
KEYWORDS_PATH = Path(__file__).resolve().parent / "data" / "lang_keywords.json"
FEEDBACK_PATH = Path(__file__).resolve().parent / "data" / "feedback.json"

# =============================
# Enhanced logging with sentiment tracking
# =============================
def log_interaction(user_text, detected_lang, translated_input, predicted_tag, bot_reply, confidence=0.0, feedback=None):
    """Enhanced logging with confidence scores and feedback"""
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_text": user_text,
            "detected_lang": detected_lang,
            "translated_input": translated_input,
            "predicted_tag": predicted_tag,
            "bot_reply": bot_reply,
            "confidence": confidence,
            "session_id": st.session_state.get("session_id", "unknown"),
            "feedback": feedback
        }
        
        logs.append(log_entry)

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save log: {e}")

# =============================
# Enhanced model loading with confidence scores
# =============================
@st.cache_resource
def load_model_and_data():
    """Load model and data with caching"""
    clf = joblib.load(MODEL_PATH)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    responses = {}
    for intent in data["intents"]:
        tag = intent.get("tag") or intent.get("intent")
        responses[tag] = intent.get("responses", [])
    
    return clf, responses

clf, responses = load_model_and_data()

# =============================
# Session initialization
# =============================
def init_session():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "user_satisfaction" not in st.session_state:
        st.session_state.user_satisfaction = []
    
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

# =============================
# Enhanced language detection
# =============================
def detect_supported_lang(text):
    """Enhanced language detection with confidence"""
    if KEYWORDS_PATH.exists():
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            lang_keywords = json.load(f)
    else:
        lang_keywords = {"ms": [], "zh-CN": []}
    
    t = text.lower()
    
    # Keyword-based detection first
    ms_score = sum(1 for word in lang_keywords.get("ms", []) if word in t)
    zh_score = sum(1 for word in text if word in lang_keywords.get("zh-CN", []))
    
    if ms_score > 0:
        return "ms", ms_score / len(text.split())
    if zh_score > 0:
        return "zh-CN", zh_score / len(text)
    
    # Fallback to langdetect
    try:
        detected = detect(text)
        confidence = 0.8  # Default confidence for langdetect
        
        if detected in ["en"]:
            return "en", confidence
        elif detected in ["ms", "id"]:
            return "ms", confidence
        elif detected in ["zh", "zh-cn", "zh-tw"]:
            return "zh-CN", confidence
        else:
            return "en", 0.5  # Low confidence fallback
    except:
        return "en", 0.3

# =============================
# Context-aware response generation
# =============================
def get_contextual_response(tag, user_text, conversation_history):
    """Generate context-aware responses"""
    base_responses = responses.get(tag, responses.get("fallback", ["Sorry, I didn't understand that."]))
    
    # Add context-specific modifications
    if tag == "greeting" and len(conversation_history) > 2:
        base_responses = ["Welcome back! How can I help you today?", "Hello again! What would you like to know?"]
    
    # Add follow-up suggestions based on intent
    follow_ups = {
        "admissions_requirements": "\n\n💡 You might also want to ask about tuition fees or scholarship opportunities.",
        "tuition_fees": "\n\n💡 Don't forget to check our scholarship programs!",
        "scholarship": "\n\n💡 Would you like to know about the application deadlines?",
        "exam_schedule": "\n\n💡 Need help with library hours for studying?",
    }
    
    response = random.choice(base_responses)
    if tag in follow_ups:
        response += follow_ups[tag]
    
    return response

# =============================
# Enhanced bot reply with confidence scoring
# =============================
def bot_reply(user_text):
    """Enhanced bot reply with confidence and context"""
    detected_lang, lang_confidence = detect_supported_lang(user_text)
    
    # Translate input if needed
    if detected_lang != "en":
        try:
            translated_input = GoogleTranslator(source="auto", target="en").translate(user_text)
        except:
            translated_input = user_text  # Fallback
    else:
        translated_input = user_text

    # Get prediction with probability scores
    try:
        tag = clf.predict([translated_input.lower()])[0]
        # Get confidence score
        if hasattr(clf, "decision_function"):
            confidence_scores = clf.decision_function([translated_input.lower()])[0]
            confidence = max(confidence_scores) if len(confidence_scores) > 1 else confidence_scores[0]
            confidence = 1 / (1 + abs(confidence))  # Convert to 0-1 range
        else:
            confidence = 0.7  # Default confidence
    except Exception:
        tag = "fallback"
        confidence = 0.1

    # Generate contextual response
    reply_en = get_contextual_response(tag, user_text, st.session_state.conversation_context)

    # Translate back if needed
    if detected_lang != "en":
        try:
            reply = GoogleTranslator(source="en", target=detected_lang).translate(reply_en)
        except:
            reply = reply_en  # Fallback
    else:
        reply = reply_en

    # Update conversation context
    st.session_state.conversation_context.append({
        "user": user_text,
        "bot": reply,
        "intent": tag,
        "confidence": confidence
    })

    # Keep only last 5 interactions for context
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context.pop(0)

    # Save to UI history
    lang_indicator = {"en": "🇺🇸 English", "ms": "🇲🇾 Malay", "zh-CN": "🇨🇳 Chinese"}
    confidence_indicator = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    
    st.session_state.history.append(("You", f"{user_text}\n\n_{lang_indicator.get(detected_lang, 'Unknown')}_"))
    st.session_state.history.append(("Bot", f"{reply}\n\n_{confidence_indicator} Confidence: {confidence:.2f}_"))

    # Log interaction
    log_interaction(user_text, detected_lang, translated_input, tag, reply, confidence)

    return confidence

# =============================
# Feedback system
# =============================
def save_feedback(rating, comment=""):
    """Save user feedback"""
    try:
        if FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "rating": rating,
            "comment": comment,
            "conversation_length": len(st.session_state.history)
        }
        
        feedback_data.append(feedback_entry)
        
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        
        st.success("Thank you for your feedback! 🙏")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# =============================
# Analytics Dashboard (Admin)
# =============================
def show_analytics():
    """Display analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    if not LOG_PATH.exists():
        st.warning("No conversation data available yet.")
        return
    
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        if not logs:
            st.warning("No conversation data available yet.")
            return
        
        df = pd.DataFrame(logs)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Conversations", len(df))
        with col2:
            st.metric("Unique Sessions", df['session_id'].nunique() if 'session_id' in df else "N/A")
        with col3:
            avg_confidence = df['confidence'].mean() if 'confidence' in df else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        with col4:
            st.metric("Languages Used", df['detected_lang'].nunique() if 'detected_lang' in df else "N/A")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Intent distribution
            if 'predicted_tag' in df:
                intent_counts = df['predicted_tag'].value_counts()
                fig = px.bar(x=intent_counts.index, y=intent_counts.values, 
                           title="Most Common Questions")
                fig.update_layout(xaxis_title="Intent", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Language distribution
            if 'detected_lang' in df:
                lang_counts = df['detected_lang'].value_counts()
                fig = px.pie(values=lang_counts.values, names=lang_counts.index, 
                           title="Language Usage")
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent conversations
        st.subheader("Recent Conversations")
        recent_df = df.tail(10)[['timestamp', 'user_text', 'predicted_tag', 'confidence']].copy()
        if 'confidence' in recent_df:
            recent_df['confidence'] = recent_df['confidence'].round(2)
        st.dataframe(recent_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# =============================
# Main App Configuration
# =============================
st.set_page_config(
    page_title="🎓 University FAQ Chatbot", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session
init_session()

# Header with logo
logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
col1, col2 = st.columns([1, 4])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=100)
with col2:
    st.title("🎓 University FAQ Chatbot")
    st.caption("Multilingual support: English • Malay • 中文")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.info(
        "This AI chatbot helps answer questions about:\n\n"
        "• 📚 Admissions & Requirements\n"
        "• 💰 Tuition Fees & Scholarships\n"
        "• 📅 Exam Schedules\n"
        "• 📖 Library Services\n"
        "• 🏠 Housing & Hostels\n"
        "• ⏰ Office Hours"
    )
    
    st.header("🔧 Session Info")
    st.text(f"Session ID: {st.session_state.session_id}")
    st.text(f"Messages: {len(st.session_state.history)}")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.conversation_context = []
        st.rerun()
    
    # Admin panel toggle
    if st.checkbox("📊 Show Analytics", help="Admin dashboard"):
        st.session_state.show_analytics = True
    else:
        st.session_state.show_analytics = False

# Main content area
main_col, feedback_col = st.columns([3, 1])

with main_col:
    # Show analytics or chat interface
    if st.session_state.get("show_analytics", False):
        show_analytics()
    else:
        # Quick action buttons
        st.subheader("🔍 Quick Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📚 Admission Requirements", use_container_width=True):
                bot_reply("what are the admission requirements")
        with col2:
            if st.button("💰 Tuition Fees", use_container_width=True):
                bot_reply("how much is the tuition fee")
        with col3:
            if st.button("📅 Exam Dates", use_container_width=True):
                bot_reply("when are the exams")
        
        # Chat input
        if user_input := st.chat_input("Ask me anything about the university..."):
            confidence = bot_reply(user_input)
        
        # Display chat history with better styling
        st.subheader("💬 Conversation")
        for i, (speaker, msg) in enumerate(st.session_state.history):
            if speaker == "You":
                with st.chat_message("user"):
                    st.write(msg)
            else:
                with st.chat_message("assistant"):
                    st.write(msg)

# Feedback panel
with feedback_col:
    if len(st.session_state.history) > 0:
        st.subheader("📝 Feedback")
        st.write("How helpful was this conversation?")
        
        rating = st.select_slider(
            "Rating",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            key=f"rating_{len(st.session_state.history)}"
        )
        
        comment = st.text_area(
            "Additional comments (optional):",
            placeholder="Any suggestions for improvement?",
            key=f"comment_{len(st.session_state.history)}"
        )
        
        if st.button("Submit Feedback"):
            save_feedback(rating, comment)

# Enhanced CSS styling
st.markdown("""
<style>
.stChatMessage {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
}

.metric-container {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}

.quick-btn {
    background: linear-gradient(45deg, #2196F3, #21CBF3);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.quick-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)
