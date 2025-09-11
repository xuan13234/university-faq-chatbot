import os
import csv
import json
import random
import re
from datetime import datetime, timedelta
from collections import deque
import time
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# Set a consistent font for matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

# ------------------------
# Optional heavy libraries with better error handling
# ------------------------
HAS_TORCH = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

HAS_SBERT = True
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    HAS_SBERT = False
    SentenceTransformer = None
    util = None

HAS_LANGDETECT = True
try:
    from langdetect import detect, DetectorFactory
    # Ensure consistent language detection
    DetectorFactory.seed = 0
except ImportError:
    HAS_LANGDETECT = False
    detect = None

HAS_DEEP_TRANSLATOR = True
try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
except ImportError:
    HAS_DEEP_TRANSLATOR = False
    DeepGoogleTranslator = None

HAS_GOOGLETRANS = True
try:
    from googletrans import Translator as GoogleTranslator
except ImportError:
    HAS_GOOGLETRANS = False
    GoogleTranslator = None

HAS_SPACY = True
nlp = None
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Model not found, we'll handle this later
        nlp = None
    except Exception:
        nlp = None
except ImportError:
    HAS_SPACY = False

HAS_SPEECH = True
try:
    import speech_recognition as sr
    import pyttsx3
except ImportError:
    HAS_SPEECH = False

HAS_PLOTLY = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    HAS_PLOTLY = False
    px = None
    go = None
    make_subplots = None

# ------------------------
# Config / filenames with path validation
# ------------------------
APP_TITLE = "🎓 University FAQ Chatbot"
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "chatbot_logs.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.csv")
FAQ_FILE = os.path.join(DATA_DIR, "faq.csv")
INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")
DATA_PTH = os.path.join(DATA_DIR, "data.pth")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CONTEXT = 5
MAX_SENT_LEN = 16
SIM_THRESHOLD = 0.62
PROB_THRESHOLD = 0.70

# University-specific information
UNIVERSITY_INFO = {
    "name": "University of Technology",
    "email": "admissions@university-tech.edu",
    "phone": "+1 (555) 123-4567",
    "address": "123 Education Boulevard, Tech City, TC 12345",
    "hours": "Monday-Friday: 8:00 AM - 6:00 PM, Saturday: 9:00 AM - 1:00 PM",
    "departments": ["Computer Science", "Engineering", "Business", "Arts", "Sciences"],
    "semester_dates": {
        "fall": "August 26 - December 15",
        "spring": "January 15 - May 10",
        "summer": "June 1 - July 31"
    }
}

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------
# Custom CSS for styling
# ------------------------
def inject_custom_css():
    st.markdown(f"""
    <style>
    /* Main container */
    .main {{
        background-color: #f8f9fa;
    }}
    
    /* University color theme */
    :root {{
        --primary-color: #1a4f8b;
        --secondary-color: #e6af21;
        --accent-color: #7d3c98;
        --light-bg: #f0f2f6;
    }}
    
    /* Chat containers */
    .user-message {{
        background: linear-gradient(135deg, var(--primary-color), #2c6db3);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        position: relative;
    }}
    
    .bot-message {{
        background: linear-gradient(135deg, #e0e0e0, #f5f5f5);
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        position: relative;
        border-left: 4px solid var(--secondary-color);
    }}
    
    /* Message metadata */
    .message-meta {{
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 4px;
    }}
    
    /* Buttons */
    .stButton button {{
        border-radius: 20px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        background: linear-gradient(135deg, var(--primary-color), #2c6db3);
        color: white;
        border: none;
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Sidebar */
    .css-1d391kg {{
        background-color: var(--light-bg);
        padding: 20px;
        border-radius: 10px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--light-bg);
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary-color);
        color: white;
    }}
    
    /* Metrics */
    .stMetric {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 4px solid var(--primary-color);
    }}
    
    /* Voice button */
    .voice-btn {{
        background: linear-gradient(135deg, var(--secondary-color), #f1c40f);
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
    }}
    
    .voice-btn:hover {{
        transform: scale(1.1);
    }}
    
    /* Feedback buttons */
    .feedback-btn {{
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0 5px;
        transition: all 0.3s ease;
    }}
    
    .feedback-btn:hover {{
        transform: scale(1.2);
    }}
    
    /* Status indicators */
    .status-indicator {{
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }}
    
    .status-online {{
        background-color: #4CAF50;
    }}
    
    .status-offline {{
        background-color: #f44336;
    }}
    
    /* Custom cards */
    .custom-card {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 4px solid var(--primary-color);
        height: fit-content;
    }}
    
    /* Animation for new messages */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.3s ease-in-out;
    }}
    
    /* Typing indicator */
    .typing-indicator {{
        background-color: #E6E7ED;
        padding: 10px 15px;
        border-radius: 18px;
        display: table;
        margin: 8px 0;
        position: relative;
    }}
    
    .typing-indicator span {{
        height: 8px;
        width: 8px;
        float: left;
        margin: 0 1px;
        background-color: #9E9EA1;
        display: block;
        border-radius: 50%;
        opacity: 0.4;
    }}
    
    .typing-indicator span:nth-of-type(1) {{
        animation: typing 1s infinite;
    }}
    
    .typing-indicator span:nth-of-type(2) {{
        animation: typing 1s 0.33s infinite;
    }}
    
    .typing-indicator span:nth-of-type(3) {{
        animation: typing 1s 0.66s infinite;
    }}
    
    @keyframes typing {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-5px); opacity: 1; }}
    }}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .user-message, .bot-message {{
            max-width: 90%;
        }}
    }}
    
    /* Evaluation charts */
    .evaluation-chart {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}
    
    /* Suggested questions */
    .suggested-question {{
        display: inline-block;
        background: linear-gradient(135deg, var(--secondary-color), #f1c40f);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .suggested-question:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}
    
    /* Context memory display */
    .context-memory {{
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin-bottom: 15px;
        font-size: 0.9rem;
    }}
    
    /* Quick response buttons */
    .quick-response {{
        background: linear-gradient(135deg, var(--primary-color), #2c6db3);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: inline-block;
        text-align: center;
    }}
    
    .quick-response:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #153e6b, #1f5ba3);
    }}
    
    /* Input area styling */
    .stTextInput>div>div>input {{
        border-radius: 20px;
        padding: 12px 16px;
    }}
    
    /* Loading spinner */
    .loading-spinner {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
        margin-right: 10px;
    }}
    
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    
    /* University header */
    .university-header {{
        background: linear-gradient(135deg, var(--primary-color), #2c6db3);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }}
    
    /* History message styling */
    .history-message {{
        padding: 10px 15px;
        border-radius: 10px;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .history-user {{
        background-color: #e6f2ff;
        border-left: 4px solid var(--primary-color);
    }}
    
    .history-bot {{
        background-color: #f0f0f0;
        border-left: 4px solid var(--secondary-color);
    }}
    
    /* Improved tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: #e9ecef;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        margin: 0 2px;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: var(--primary-color);
        color: white;
    }}
    
    /* Improved selectbox styling */
    .stSelectbox [data-baseweb="select"] {{
        border-radius: 10px;
    }}
    
    /* Improved metric cards */
    .metric-card {{
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 15px;
    }}
    
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: var(--primary-color);
    }}
    
    .metric-label {{
        font-size: 14px;
        color: #6c757d;
    }}
    
    /* University info cards */
    .info-card {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 4px solid var(--primary-color);
        height: auto;
        min-height: 200px;
    }}
    
    .info-card h3 {{
        color: var(--primary-color);
        margin-top: 0;
    }}
    
    .info-card ul {{
        padding-left: 20px;
    }}
    
    .info-card a {{
        color: var(--primary-color);
        text-decoration: none;
    }}
    
    .info-card a:hover {{
        text-decoration: underline;
    }}
    
    /* Settings panel */
    .settings-panel {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------------
# Safe CSV init with error handling
# ------------------------
def ensure_csv(path, header):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, quoting=csv.QUOTE_ALL)
                w.writerow(header)
    except Exception as e:
        st.sidebar.error(f"Failed to create {path}: {e}")

ensure_csv(LOG_FILE, ["timestamp", "user_input", "user_lang", "translated_input", "predicted_tag", "response", "feedback", "confidence", "detected_lang", "translated_from"])
ensure_csv(HISTORY_FILE, ["timestamp", "speaker", "message"])
ensure_csv(os.path.join(DATA_DIR, "ratings.csv"), ["timestamp", "rating"])

# ------------------------
# Load intents & optional FAQ with error handling
# ------------------------
def load_intents():
    try:
        if os.path.exists(INTENTS_FILE):
            with open(INTENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Create university-specific intents
            university_intents = {
                "intents": [
                    {
                        "tag": "greeting",
                        "patterns": [
                            "Hello", "Hi", "Hey", "How are you", "Good day", 
                            "Good morning", "Good afternoon", "Good evening"
                        ],
                        "responses": [
                            "Hello! Welcome to University of Technology. How can I assist you today?",
                            "Hi there! I'm here to help with any questions about our university.",
                            "Greetings! How can I help you with your university inquiries today?"
                        ]
                    },
                    {
                        "tag": "goodbye",
                        "patterns": [
                            "Bye", "See you later", "Goodbye", "Take care", 
                            "That's all", "Thank you, goodbye", "I have to go"
                        ],
                        "responses": [
                            "Goodbye! Have a great day and don't hesitate to reach out if you have more questions!",
                            "See you later! Feel free to come back if you need more information about our university.",
                            "Take care! Remember to check our website for the latest university updates."
                        ]
                    },
                    {
                        "tag": "admissions",
                        "patterns": [
                            "How do I apply", "Admission requirements", "Application process",
                            "What are the entry requirements", "How to get admitted",
                            "Application deadline", "When should I apply", "Admission criteria"
                        ],
                        "responses": [
                            "The application process involves submitting an online application, academic transcripts, test scores, and a personal statement. The deadline for Fall admission is typically January 15th.",
                            "Admission requirements vary by program but generally include a high school diploma with a minimum GPA of 3.0, SAT/ACT scores, and letters of recommendation.",
                            "You can apply through our online portal. Required documents include transcripts, test scores, a personal essay, and two letters of recommendation."
                        ]
                    },
                    {
                        "tag": "programs",
                        "patterns": [
                            "What programs do you offer", "Available majors", "List of courses",
                            "Degree programs", "What can I study", "Academic programs",
                            "Graduate programs", "Undergraduate programs"
                        ],
                        "responses": [
                            f"We offer a wide range of programs across our {len(UNIVERSITY_INFO['departments'])} departments: {', '.join(UNIVERSITY_INFO['departments'])}.",
                            "Our university provides both undergraduate and graduate programs in various fields including technology, business, arts, and sciences.",
                            "You can explore our full program catalog on our website, which includes bachelor's, master's, and doctoral degrees."
                        ]
                    },
                    {
                        "tag": "tuition",
                        "patterns": [
                            "How much is tuition", "Tuition fees", "Cost of attendance",
                            "What are the fees", "How much does it cost", "Tuition and fees",
                            "Financial information", "Cost per credit"
                        ],
                        "responses": [
                            "Undergraduate tuition is $15,000 per semester for full-time students. Graduate tuition varies by program but averages $20,000 per semester.",
                            "Tuition costs depend on your program and enrollment status. For detailed information, please visit our tuition and fees page on the university website.",
                            "We offer transparent pricing for all programs. Full-time undergraduate tuition is $15,000/semester, with additional fees for specific programs or courses."
                        ]
                    },
                    {
                        "tag": "scholarships",
                        "patterns": [
                            "Scholarship opportunities", "Financial aid", "How to get scholarship",
                            "Merit scholarships", "Need-based aid", "Scholarship application",
                            "Grants", "Work-study programs"
                        ],
                        "responses": [
                            "We offer various scholarships including merit-based, need-based, and program-specific awards. The application deadline for most scholarships is March 1st.",
                            "Financial aid options include scholarships, grants, loans, and work-study programs. Complete the FAFSA to be considered for need-based aid.",
                            "Merit scholarships are available for students with outstanding academic achievements. Additional scholarships are offered for specific majors and backgrounds."
                        ]
                    },
                    {
                        "tag": "housing",
                        "patterns": [
                            "On-campus housing", "Dormitories", "Student housing",
                            "Living on campus", "Residence halls", "Housing options",
                            "Room and board", "Housing application"
                        ],
                        "responses": [
                            "We offer several on-campus housing options including traditional dormitories, suite-style living, and apartment-style residences. The housing application opens on April 1st.",
                            "First-year students are guaranteed housing in our residence halls. Options include single, double, and triple occupancy rooms with various meal plan options.",
                            "On-campus housing provides a convenient living experience with amenities like WiFi, laundry facilities, and community spaces. Applications are accepted starting April 1st."
                        ]
                    },
                    {
                        "tag": "campus_tour",
                        "patterns": [
                            "Campus visit", "Schedule a tour", "Visit the university",
                            "Campus tour", "Open house", "Information session",
                            "Tour the campus", "See the campus"
                        ],
                        "responses": [
                            "You can schedule a campus tour through our admissions website. We offer both in-person and virtual tour options.",
                            "Campus tours are available Monday through Friday at 10 AM and 2 PM. You can register online or by calling our admissions office.",
                            "We'd love to show you around! Schedule a campus tour on our website to see our facilities and learn more about student life."
                        ]
                    },
                    {
                        "tag": "faculty",
                        "patterns": [
                            "Who are the professors", "Faculty information", "Teaching staff",
                            "Professor contacts", "Department faculty", "Who teaches",
                            "Faculty directory", "Find a professor"
                        ],
                        "responses": [
                            "Our faculty includes renowned experts in their fields. You can browse faculty profiles by department on our university website.",
                            "Each department has its own faculty page with information about professors, their research interests, and contact information.",
                            "Our faculty directory is available online where you can search for professors by name, department, or research area."
                        ]
                    },
                    {
                        "tag": "deadlines",
                        "patterns": [
                            "Application deadline", "When is the deadline", "Important dates",
                            "Registration deadline", "Tuition due date", "Semester dates",
                            "Academic calendar", "Term dates"
                        ],
                        "responses": [
                            f"The academic calendar includes: Fall Semester: {UNIVERSITY_INFO['semester_dates']['fall']}, Spring Semester: {UNIVERSITY_INFO['semester_dates']['spring']}, Summer Semester: {UNIVERSITY_INFO['semester_dates']['summer']}.",
                            "Application deadlines vary by program. For undergraduate programs, the priority deadline is January 15th for Fall admission.",
                            "Important dates including registration periods, add/drop deadlines, and exam schedules are available on the academic calendar on our website."
                        ]
                    },
                    {
                        "tag": "contact",
                        "patterns": [
                            "How to contact", "Phone number", "Email address",
                            "Where are you located", "Office location", "Admissions office",
                            "Contact information", "Get in touch"
                        ],
                        "responses": [
                            f"You can reach us at {UNIVERSITY_INFO['phone']} or {UNIVERSITY_INFO['email']}. Our address is {UNIVERSITY_INFO['address']}.",
                            f"The admissions office is open {UNIVERSITY_INFO['hours']}. You can call {UNIVERSITY_INFO['phone']} or email {UNIVERSITY_INFO['email']}.",
                            f"Contact information: Phone: {UNIVERSITY_INFO['phone']}, Email: {UNIVERSITY_INFO['email']}, Address: {UNIVERSITY_INFO['address']}. Office hours: {UNIVERSITY_INFO['hours']}."
                        ]
                    },
                    {
                        "tag": "library",
                        "patterns": [
                            "Library hours", "Study spaces", "Research resources",
                            "Library database", "Borrow books", "Library services",
                            "Study rooms", "Online resources"
                        ],
                        "responses": [
                            "The university library is open Monday-Thursday 7:30 AM - 11:00 PM, Friday 7:30 AM - 8:00 PM, Saturday 10:00 AM - 6:00 PM, and Sunday 12:00 PM - 10:00 PM.",
                            "Our library offers extensive digital resources, study spaces, research assistance, and borrowing services. You can access online databases 24/7 with your student credentials.",
                            "The library provides study rooms, computer labs, research assistance, and access to thousands of journals and databases. Current hours and services are listed on the library website."
                        ]
                    }
                ]
            }
            with open(INTENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(university_intents, f, indent=2)
            return university_intents
    except Exception as e:
        st.sidebar.error(f"Error loading intents: {e}")
        return {"intents": []}

intents = load_intents()

def load_faq():
    try:
        if os.path.exists(FAQ_FILE):
            return pd.read_csv(FAQ_FILE)
        else:
            # Create university-specific FAQ
            university_faq = pd.DataFrame({
                "question": [
                    "What are the admission requirements?",
                    "How much is tuition?",
                    "What programs do you offer?",
                    "How can I apply for financial aid?",
                    "What housing options are available?",
                    "How do I schedule a campus tour?",
                    "What are the application deadlines?",
                    "How do I contact the admissions office?",
                    "What library resources are available?",
                    "Are there scholarships available?",
                    "What is the student-to-faculty ratio?",
                    "What campus facilities are available?",
                    "How do I access online student portals?",
                    "What dining options are on campus?",
                    "What extracurricular activities are available?"
                ],
                "answer": [
                    "Admission requirements include a completed application, high school transcripts, SAT/ACT scores, letters of recommendation, and a personal statement. Specific requirements vary by program.",
                    "Undergraduate tuition is $15,000 per semester for full-time students. Additional fees may apply for specific programs or courses.",
                    f"We offer programs in {', '.join(UNIVERSITY_INFO['departments'])} at both undergraduate and graduate levels.",
                    "Complete the FAFSA form and our university scholarship application to be considered for financial aid. Additional program-specific scholarships may be available.",
                    "We offer traditional dormitories, suite-style living, and apartment-style residences. First-year students are guaranteed housing.",
                    "You can schedule a campus tour through our website or by calling the admissions office at " + UNIVERSITY_INFO["phone"] + ".",
                    "The priority application deadline for Fall admission is January 15th. Some programs have different deadlines, so check our website for details.",
                    "You can reach the admissions office at " + UNIVERSITY_INFO["phone"] + " or " + UNIVERSITY_INFO["email"] + ". Office hours are " + UNIVERSITY_INFO["hours"] + ".",
                    "The library offers extensive physical and digital collections, study spaces, research assistance, and access to numerous academic databases.",
                    "Yes, we offer merit-based, need-based, and program-specific scholarships. The application deadline for most scholarships is March 1st.",
                    "Our student-to-faculty ratio is 15:1, ensuring personalized attention and interaction with professors.",
                    "Campus facilities include modern classrooms, laboratories, library, recreation center, dining halls, and student union building.",
                    "You can access student portals through our website using your student ID and password. Technical support is available if you encounter issues.",
                    "Campus dining options include multiple cafeterias, coffee shops, and food courts offering varied menus including vegetarian, vegan, and allergen-free options.",
                    "We offer over 100 student clubs and organizations, intramural sports, cultural events, leadership programs, and community service opportunities."
                ]
            })
            university_faq.to_csv(FAQ_FILE, index=False)
            return university_faq
    except Exception as e:
        st.sidebar.error(f"Error loading FAQ: {e}")
        return None

faq_df = load_faq()

# ------------------------
# Embeddings (SBERT) - cached with error handling
# ------------------------
@st.cache_resource
def load_embedder(model_name=EMBED_MODEL_NAME):
    if not HAS_SBERT:
        st.sidebar.warning("Sentence Transformers not available. Some features will be limited.")
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.sidebar.error(f"Error loading embedder: {e}")
        return None

embedder = load_embedder()

# Precompute intent-pattern embeddings
intent_pattern_embeddings = []
if embedder and intents.get("intents"):
    for intent in intents.get("intents", []):
        patterns = intent.get("patterns", [])
        emb = None
        try:
            if patterns:
                emb = embedder.encode(patterns, convert_to_tensor=True)
        except Exception as e:
            st.sidebar.error(f"Error encoding patterns for {intent.get('tag', 'unknown')}: {e}")
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
    except Exception as e:
        st.sidebar.error(f"Error encoding FAQ: {e}")
        faq_embeddings = None

# ------------------------
# Translation / detection helpers with better error handling
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
# Text cleaning / lemmatization with error handling
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
# Time-related questions handler
# ------------------------
def handle_time_question(text):
    time_phrases = [
        "time", "what time", "current time", "what is the time", 
        "time now", "what's the time", "tell me the time"
    ]
    
    date_phrases = [
        "date", "what date", "current date", "what is the date",
        "date today", "what's the date", "tell me the date"
    ]
    
    day_phrases = [
        "day", "what day", "today", "what day is it", "what day is today"
    ]
    
    text_lower = text.lower()
    
    for phrase in time_phrases:
        if phrase in text_lower:
            return f"🕒 The current time is {datetime.now().strftime('%H:%M:%S')}"
    
    for phrase in date_phrases:
        if phrase in text_lower:
            return f"📅 Today's date is {datetime.now().strftime('%Y-%m-%d')}"
    
    for phrase in day_phrases:
        if phrase in text_lower:
            return f"📆 Today is {datetime.now().strftime('%A')}"
    
    return None

# ------------------------
# Semantic matchers with error handling
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
        try:
            scores = util.cos_sim(u_emb, item["emb"])[0]
            value = float(scores.max())
            if value > best_score:
                best_score = value
                best_tag = item["tag"]
                if item["responses"]:
                    best_resp = random.choice(item["responses"])
        except Exception:
            continue
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
            # Clean the pattern before comparing
            cleaned_pattern = clean_text(p)
            if cleaned_pattern and cleaned_pattern in t:
                return intent.get("tag"), 0.5, random.choice(intent.get("responses", ["I can help with that."]))
    return None, 0.0, None

# ------------------------
# Optional PyTorch model loading with error handling
# ------------------------
model = None
word2idx = {}
tags = []
if HAS_TORCH and os.path.exists(DATA_PTH):
    try:
        data = torch.load(DATA_PTH, map_location=torch.device("cpu"))
        word2idx = data.get("word2idx", {})
        tags = data.get("tags", [])
        
        # Simplified model loading without dynamic import
        class SimpleChatbot(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
                super(SimpleChatbot, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim)
                self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        # ... (previous code continues)

        try:
            model = SimpleChatbot(data["vocab_size"], data["embed_dim"], data["hidden_size"], len(tags))
            model.load_state_dict(data["model_state"])
            model.eval()
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            model = None
    except Exception as e:
        st.sidebar.error(f"Error loading model data: {e}")
        model = None

def model_predict_intent(text):
    if model is None or not word2idx:
        return None, 0.0
    try:
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
    except Exception:
        return None, 0.0

# ------------------------
# NER (spaCy) with error handling
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
# spaCy model download with error handling
# ------------------------
def download_spacy_model():
    """Download spaCy model if not available"""
    try:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to download spaCy model: {e}")
        return False

# ------------------------
# Enhanced text-to-speech with better error handling
# ------------------------
def speak_text(text):
    if not HAS_SPEECH:
        return False
    
    try:
        # Try to initialize the engine
        engine = pyttsx3.init()
        
        # Set properties (optional)
        engine.setProperty('rate', 150)  # Speed percent
        engine.setProperty('volume', 0.9)  # Volume 0-1
        
        # Try to set a more natural voice if available
        try:
            voices = engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if "female" in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
        except:
            pass  # If voice setting fails, continue with default
        
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        if "espeak" in str(e).lower():
            st.sidebar.warning(
                "Text-to-speech requires eSpeak. "
                "On Ubuntu/Debian: sudo apt-get install espeak\n"
                "On macOS: brew install espeak\n"
                "On Windows: Download from http://espeak.sourceforge.net"
            )
        else:
            st.sidebar.error(f"Text-to-speech error: {str(e)}")
        return False

# ------------------------
# Plotly fallback with matplotlib
# ------------------------
def create_visualization(data, viz_type, title, x_label, y_label, **kwargs):
    """
    Create visualization with Plotly or fallback to matplotlib
    """
    if HAS_PLOTLY:
        try:
            if viz_type == "line":
                fig = px.line(data, x=data.index, y=data.values, title=title,
                             labels={"x": x_label, "y": y_label})
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=400
                )
                return fig
            elif viz_type == "bar":
                fig = px.bar(data, x=data.index, y=data.values, title=title,
                            labels={"x": x_label, "y": y_label})
                fig.update_layout(height=400)
                return fig
            elif viz_type == "pie":
                fig = px.pie(data, values=data.values, names=data.index, title=title)
                fig.update_layout(height=400)
                return fig
            elif viz_type == "histogram":
                fig = px.histogram(data, x=data.values, title=title,
                                  labels={"x": x_label, "y": y_label})
                fig.update_layout(bargap=0.1, height=400)
                return fig
        except Exception as e:
            st.sidebar.error(f"Plotly error: {e}. Falling back to matplotlib.")
    
    # Fallback to matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    if viz_type == "line":
        ax.plot(data.index, data.values)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45)
    elif viz_type == "bar":
        ax.bar(data.index, data.values)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45)
    elif viz_type == "pie":
        ax.pie(data.values, labels=data.index, autopct='%1.1f%%')
        ax.set_title(title)
    elif viz_type == "histogram":
        ax.hist(data.values, bins=20)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
    return fig

# ------------------------
# University-specific booking system
# ------------------------
def handle_university_booking(user_input, current_state=None):
    """
    Handle university-specific booking process (tours, appointments, etc.)
    """
    if current_state is None:
        current_state = {}
    
    # Extract booking type using NER
    booking_types = ["tour", "appointment", "consultation", "visit", "information session"]
    extracted_type = None
    
    # Try to extract using spaCy NER
    if HAS_SPACY and nlp:
        doc = nlp(user_input)
        for ent in doc.ents:
            if ent.label_ in ["EVENT", "ORG"] or any(bt in ent.text.lower() for bt in booking_types):
                extracted_type = ent.text
                break
    
    # Fallback: extract after booking keywords
    if not extracted_type:
        booking_patterns = ["book a", "schedule a", "i want to", "i need a"]
        for pattern in booking_patterns:
            if pattern in user_input.lower():
                parts = user_input.lower().split(pattern)
                if len(parts) > 1:
                    extracted_type = parts[1].strip().split(" ")[0]
                    break
    
    # Determine current step in booking process
    if "type" not in current_state:
        # First step - determine booking type
        booking_type = extracted_type or "tour"
        current_state["type"] = booking_type
        
        if extracted_type:
            return f"🔎 I'll help you schedule a {booking_type}. What date would work best for you?", current_state, False
        else:
            return "🔎 I'll help you with scheduling. What would you like to book? (tour, appointment, etc.)", current_state, False
    
    elif "type" in current_state and "date" not in current_state:
        # Second step - get date
        current_state["date"] = user_input
        return "📅 Thank you. What time would you prefer?", current_state, False
    
    elif "date" in current_state and "time" not in current_state:
        # Third step - get time
        current_state["time"] = user_input
        return "🕒 Great. Could you please provide your name and contact information?", current_state, False
    
    elif "time" in current_state and "contact" not in current_state:
        # Fourth step - get contact info
        current_state["contact"] = user_input
        booking_type = current_state["type"]
        date = current_state["date"]
        time = current_state["time"]
        contact = current_state["contact"]
        
        # Save booking to a simple CSV file
        try:
            booking_file = os.path.join(DATA_DIR, "university_bookings.csv")
            if not os.path.exists(booking_file):
                with open(booking_file, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["timestamp", "type", "date", "time", "contact"])
            
            with open(booking_file, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), booking_type, date, time, contact])
        except Exception as e:
            st.sidebar.error(f"Error saving booking: {e}")
        
        return f"✅ Your {booking_type} has been scheduled for {date} at {time}. We will contact you at {contact} to confirm. Thank you!", current_state, True
    
    return "I'm not sure how to process your booking request. Please try again.", current_state, True

# ------------------------
# University-specific recommendation system
# ------------------------
def generate_university_recommendation(user_input, context):
    """
    Generate university-specific recommendations
    """
    # Analyze context for program/department mentions
    program_types = UNIVERSITY_INFO["departments"] + ["program", "major", "course", "degree"]
    detected_programs = []
    
    # Check current message and context for program mentions
    all_text = [user_input] + list(context)
    
    for text in all_text:
        if HAS_SPACY and nlp:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"] or any(pt in ent.text.lower() for pt in program_types):
                    detected_programs.append(ent.text)
    
    if detected_programs:
        program = detected_programs[-1]  # Most recent program mentioned
        
        # University-specific recommendation logic
        if "computer" in program.lower() or "tech" in program.lower():
            return "📌 Based on your interest in technology, I recommend exploring our Computer Science department. We offer cutting-edge programs in AI, cybersecurity, and software engineering."
        elif "engineer" in program.lower():
            return "📌 Our Engineering programs are highly regarded, with specializations in mechanical, electrical, and civil engineering. We also offer a unique biomedical engineering track."
        elif "business" in program.lower():
            return "📌 For business-minded students, our Business School offers excellent programs in management, finance, and marketing, with opportunities for internships with leading companies."
        elif "art" in program.lower() or "design" in program.lower():
            return "📌 Our Arts programs provide creative students with opportunities in visual arts, performing arts, and digital media design. We have state-of-the-art studios and exhibition spaces."
        elif "science" in program.lower():
            return "📌 Our Sciences department offers rigorous programs in biology, chemistry, physics, and environmental science, with extensive research opportunities for undergraduates."
        else:
            return f"📌 I recommend exploring our {program} program further. You can schedule a department visit or speak with a current student in that program."
    else:
        # Generic university recommendations
        recommendations = [
            "📌 I recommend scheduling a campus tour to get a feel for our university community and facilities.",
            "📌 Based on popular choices, many students find our Computer Science and Business programs to be excellent choices with great career outcomes.",
            "📌 Consider our honors program if you're looking for a challenging academic experience with smaller class sizes and research opportunities.",
            "📌 I'd recommend exploring our study abroad options - many students find this to be a transformative experience during their university years."
        ]
        return random.choice(recommendations)

# ------------------------
# University-specific troubleshooting
# ------------------------
def provide_university_troubleshooting(user_input):
    """
    Provide university-specific troubleshooting assistance
    """
    # Extract issue type
    issue_types = ["login", "portal", "email", "password", "registration", "course", "technical", "wifi", "library"]
    detected_issue = None
    
    if HAS_SPACY and nlp:
        doc = nlp(user_input)
        for ent in doc.ents:
            if any(it in ent.text.lower() for it in issue_types):
                detected_issue = ent.text
                break
    
    # University-specific troubleshooting knowledge base
    troubleshooting_kb = {
        "login": [
            "🔧 If you're having trouble logging into student portals, try resetting your password using the 'Forgot Password' link.",
            "🔧 Login issues are often resolved by clearing your browser cache or trying a different browser."
        ],
        "portal": [
            "🔧 The student portal is maintained by our IT department. If you're experiencing issues, contact the IT help desk at it-support@university-tech.edu.",
            "🔧 Portal issues can sometimes be resolved by logging out completely, clearing browser cookies, and logging back in."
        ],
        "email": [
            "🔧 For university email issues, contact our IT support team at it-support@university-tech.edu or call (555) 123-HELP.",
            "🔧 Email setup instructions are available on our IT website. Make sure you're using the correct server settings."
        ],
        "password": [
            "🔧 You can reset your password using the 'Forgot Password' link on the login page. You'll need your student ID and birthdate to verify identity.",
            "🔧 Password resets can be done through our identity management system. If you're still having issues, contact the IT help desk."
        ],
        "registration": [
            "🔧 Course registration issues are handled by the registrar's office. Contact them at registrar@university-tech.edu for assistance.",
            "🔧 If you're having trouble registering for courses, it might be due to prerequisites, holds on your account, or class capacity issues."
        ],
        "wifi": [
            "🔧 For WiFi connectivity issues, make sure you're using the correct network (Eduroam) and your login credentials.",
            "🔧 WiFi setup instructions are available on our IT website. If you continue to have issues, visit the IT help desk in the library."
        ]
    }
    
    if detected_issue:
        for key in troubleshooting_kb:
            if key in detected_issue.lower():
                return random.choice(troubleshooting_kb[key]) + " If this doesn't resolve your issue, please contact the appropriate support department."
    
    # Default university troubleshooting advice
    default_advice = [
        "🛠️ For technical issues, please contact our IT support team at it-support@university-tech.edu or (555) 123-HELP.",
        "🛠️ Many common issues are addressed in our student knowledge base. You can access it through the student portal.",
        "🛠️ If you're experiencing difficulties, please reach out to the relevant department directly for assistance."
    ]
    return random.choice(default_advice)

def evaluate_chatbot(log_file="data/chatbot_logs.csv", output_file="chatbot_evaluation.csv"):
    try:
        import csv
        import numpy as np

        # Read the file with proper CSV handling
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 0 and ',' in lines[0] and lines[0].count(',') > 5:
                reader = csv.reader(lines)
                data = []
                for row in reader:
                    if len(row) > 0 and ',' in row[0]:
                        split_row = row[0].split(',')
                        if len(row) > 1:
                            split_row.extend(row[1:])
                        data.append(split_row)
                    else:
                        data.append(row)
                if len(data) > 0:
                    df = pd.DataFrame(data[1:], columns=data[0])
                else:
                    st.error("❌ No data found in the log file.")
                    return
            else:
                df = pd.read_csv(log_file, engine="python", on_bad_lines="skip")

        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if 'time' in col_lower:
                column_mapping[col] = 'timestamp'
            elif 'user' in col_lower or 'input' in col_lower:
                column_mapping[col] = 'user_input'
            elif 'tag' in col_lower or 'intent' in col_lower:
                column_mapping[col] = 'predicted_tag'
            elif 'response' in col_lower or 'answer' in col_lower:
                column_mapping[col] = 'response'
            elif 'correct' in col_lower:
                column_mapping[col] = 'correct'
            elif 'feedback' in col_lower:
                column_mapping[col] = 'feedback'
        df = df.rename(columns=column_mapping)

        # Check required columns
        if not all(col in df.columns for col in ['user_input', 'response']):
            st.error("❌ Missing essential columns in log file.")
            return

        # Handle correctness
        if 'correct' in df.columns:
            df = df.dropna(subset=["correct"])
            if not df.empty:
                df["correct"] = df["correct"].astype(int)

        # Analysis
        df['response_length'] = df['response'].apply(lambda x: len(str(x).split()))
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        for _, row in df.iterrows():
            ref = [str(row["user_input"]).split()]
            cand = str(row["response"]).split()
            try:
                bleu_scores.append(sentence_bleu(ref, cand, smoothing_function=smoothie))
            except:
                bleu_scores.append(0)
        df['bleu_score'] = bleu_scores

        analysis_results = {
            "avg_response_length": df['response_length'].mean(),
            "avg_bleu": np.mean(bleu_scores)
        }
        if 'predicted_tag' in df.columns:
            analysis_results['unique_intents'] = df['predicted_tag'].nunique()
        if 'correct' in df.columns:
            analysis_results['accuracy'] = df['correct'].mean()

        create_visualization(df, analysis_results)

        # Save CSV
        results_data = []
        for k, v in analysis_results.items():
            results_data.append({"Metric": k, "Value": v})
        pd.DataFrame(results_data).to_csv(output_file, index=False)

        st.success(f"✅ Evaluation complete. Results saved to {output_file}")
        if not df.empty:
            create_evaluation_visualizations(df, analysis_results)
    except Exception as e:
        st.error(f"❌ Evaluation error: {str(e)}")


def create_visualization(df, analysis_results):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Chatbot Conversation Analysis', fontsize=16, fontweight='bold')

    # Response length distribution
    ax1 = fig.add_subplot(221)
    ax1.hist(df['response_length'], bins=15, color='skyblue', edgecolor='black')
    ax1.set_title('Response Length Distribution')

    # BLEU score distribution
    ax2 = fig.add_subplot(222)
    ax2.hist(df['bleu_score'], bins=15, color='lightgreen', edgecolor='black')
    ax2.set_title('BLEU Score Distribution')

    # Intents
    ax3 = fig.add_subplot(223)
    if 'predicted_tag' in df.columns:
        top_intents = df['predicted_tag'].value_counts().head(5)
        ax3.barh(top_intents.index, top_intents.values, color='orange')
        ax3.set_title('Top 5 Intents')
    else:
        ax3.text(0.5, 0.5, 'No intent data available', ha='center')

    # Correctness
    ax4 = fig.add_subplot(224)
    if 'correct' in df.columns:
        df['correct'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax4)
        ax4.set_title('Response Correctness')
    else:
        ax4.text(0.5, 0.5, 'No correctness data', ha='center')

    plt.tight_layout()
    plt.savefig("chatbot_evaluation_dashboard.png", dpi=150)
    plt.close()


# ------------------------
# Special commands - University focused
# ------------------------
def special_commands(msg):
    if not msg:
        return None
    
    # Initialize booking state if not exists
    if "booking_state" not in st.session_state:
        st.session_state.booking_state = {}
    
    # Check if we're in the middle of a booking flow
    if st.session_state.get("in_booking", False):
        response, new_state, completed = handle_university_booking(msg, st.session_state.booking_state)
        st.session_state.booking_state = new_state
        
        if completed:
            st.session_state.in_booking = False
            st.session_state.booking_state = {}
        
        return ("booking", response)
    
    # Handle booking initiation
    if msg.startswith("/book") or any(word in msg.lower() for word in ["book", "schedule", "appointment", "tour", "visit"]):
        st.session_state.in_booking = True
        response, new_state, _ = handle_university_booking(msg, {})
        st.session_state.booking_state = new_state
        return ("booking", response)
    
    # Handle recommendation requests
    if msg.startswith("/recommend") or any(word in msg.lower() for word in ["recommend", "suggest", "what should", "which", "advice"]):
        context = st.session_state.get("context", deque(maxlen=MAX_CONTEXT))
        recommendation = generate_university_recommendation(msg, context)
        return ("recommendation", recommendation)
    
    # Handle troubleshooting requests
    if msg.startswith("/troubleshoot") or any(word in msg.lower() for word in ["problem", "issue", "not working", "error", "fix", "help with"]):
        troubleshooting = provide_university_troubleshooting(msg)
        return ("troubleshoot", troubleshooting)
    
    # Handle help command
    if msg.startswith("/help"):
        help_text = "🤖 University Chatbot - Available Commands:\n\n"
        help_text += "• /book - Schedule a campus tour or appointment\n"
        help_text += "• /recommend - Get program recommendations\n"
        help_text += "• /troubleshoot - Get help with technical issues\n"
        help_text += "• /clear - Clear chat history\n"
        help_text += "• /feedback - Provide feedback\n\n"
        help_text += "I can also help with these topics:\n"
        
        # Add intents to help text
        for intent in intents.get("intents", []):
            if intent.get("patterns"):
                help_text += f"• {intent['patterns'][0]}\n"
        
        return ("help", help_text)
    
    # Handle clear command
    if msg.startswith("/clear"):
        st.session_state["messages"] = []
        st.session_state["context"] = deque(maxlen=MAX_CONTEXT)
        st.session_state.in_booking = False
        st.session_state.booking_state = {}
        return ("clear", "🗑️ Chat history cleared.")
    
    # Handle feedback command
    if msg.startswith("/feedback"):
        parts = msg.split(maxsplit=1)
        feedback = parts[1] if len(parts) > 1 else ""
        if feedback:
            with open(os.path.join(DATA_DIR, "user_feedback.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {feedback}\n")
            return ("feedback", "📝 Thank you for your feedback! We appreciate your input to improve our university services.")
        else:
            return ("feedback", "📝 Please provide your feedback after the /feedback command. For example: /feedback I found the chatbot very helpful!")
    
    return None

# ------------------------
# Speech functions with error handling
# ------------------------
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
# Logging helpers with error handling
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
    except Exception as e:
        st.sidebar.error(f"Error logging interaction: {e}")

def log_history(speaker, message):
    try:
        with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speaker, message])
    except Exception as e:
        st.sidebar.error(f"Error logging history: {e}")

# ------------------------
# Process user input function
# ------------------------
def process_user_input(user_input):
    user_lang = detect_language_safe(user_input) if HAS_LANGDETECT else "en"
    translated_input, translated_from = translate_to_en(user_input, src=user_lang)

    proc_text = lemmatize_text(clean_text(translated_input))

    tag = None
    response = None
    conf = 0.0

    # First check for special commands
    sc = special_commands(user_input)
    if sc:
        tag, response = sc
        conf = 1.0
    else:
        # Check for time-related questions
        time_response = handle_time_question(user_input)
        if time_response:
            tag = "time"
            response = time_response
            conf = 1.0
        else:
            # Check FAQ
            faq_ans, faq_score = semantic_faq_match(proc_text)
            if faq_ans and faq_score >= SIM_THRESHOLD:
                tag = "faq"
                response = faq_ans
                conf = faq_score
            else:
                # Try model prediction
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

                # Try semantic matching if no match yet
                if tag is None:
                    s_tag, s_score, s_resp = semantic_intent_match(proc_text)
                    if s_tag and s_score >= SIM_THRESHOLD:
                        tag = s_tag
                        response = s_resp if s_resp else (random.choice([r for it in intents.get("intents", []) if it.get("tag") == s_tag for r in it.get("responses", [])]) if intents.get("intents") else "I can help.")
                        conf = s_score

                # Try keyword matching if no match yet
                if tag is None:
                    k_tag, k_score, k_resp = keyword_intent_match(proc_text)
                    if k_tag:
                        tag = k_tag
                        response = k_resp
                        conf = k_score

                # If all else fails, use unknown response
                if tag is None:
                    tag = "unknown"
                    # Provide more helpful unknown responses based on context
                    last_context = list(st.session_state["context"])[-1] if st.session_state["context"] else ""
                    context_based_responses = [
                        f"I'm not sure I understand. Are you asking about {last_context}?",
                        "Could you provide more details about your question?",
                        "I'm still learning about this topic. Could you try rephrasing?",
                        "That's an interesting question. Let me check my knowledge base and get back to you."
                    ] if last_context else [
                        "🤔 I'm not sure I understand. Could you rephrase that?",
                        "🔍 I'm still learning. Could you try asking in a different way?",
                        "❓ I didn't catch that. Can you provide more details?",
                        "💡 That's an interesting question. Let me check my knowledge base and get back to you."
                    ]
                    response = random.choice(context_based_responses)
                    conf = 0.0

    entities = extract_entities(proc_text)
    
    # Handle booking flow responses that might contain placeholders
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

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")
inject_custom_css()

# University header
st.markdown(f"""
<div class="university-header">
    <h1>{UNIVERSITY_INFO['name']}</h1>
    <p>Intelligent FAQ Chatbot</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)
st.sidebar.title("🎓 University Chatbot")
st.sidebar.info("Ask me about admissions, programs, scholarships, campus life, and more!")

# Check if spaCy is available but model is missing
if HAS_SPACY and nlp is None:
    st.sidebar.warning("spaCy English model not found.")
    if st.sidebar.button("Download spaCy Model"):
        if download_spacy_model():
            try:
                nlp = spacy.load("en_core_web_sm")
                st.sidebar.success("spaCy model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to load spaCy model: {e}")

# Check TTS availability
tts_available = False
if HAS_SPEECH:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(" ")
        engine.runAndWait()
        tts_available = True
    except:
        tts_available = False

# --- Sidebar: Translation selector ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Response Language")
language_options = {
    "English 🇬🇧": "en",
    "Chinese 🇨🇳": "zh-cn",
    "Spanish 🇪🇸": "es",
    "French 🇫🇷": "fr",
    "Arabic 🇦🇪": "ar",
    "Hindi 🇮🇳": "hi",
    "German 🇩🇪": "de"
}
selected_lang_display = st.sidebar.selectbox("Select language for responses:", list(language_options.keys()))
TARGET_LANG_CODE = language_options[selected_lang_display]

# Status indicators
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 System Status")
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if embedder else 'status-offline'}'></span> **SBERT Embeddings:** {'Available' if embedder else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if (HAS_DEEP_TRANSLATOR or HAS_GOOGLETRANS) else 'status-offline'}'></span> **Translation:** {'Available' if (HAS_DEEP_TRANSLATOR or HAS_GOOGLETRANS) else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if HAS_LANGDETECT else 'status-offline'}'></span> **Language Detection:** {'Available' if HAS_LANGDETECT else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if model else 'status-offline'}'></span> **PyTorch Model:** {'Loaded' if model else 'Not Loaded'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if tts_available else 'status-offline'}'></span> **Speech I/O:** {'Available' if tts_available else 'Not Available'}", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='status-indicator {'status-online' if HAS_PLOTLY else 'status-offline'}'></span> **Plotly Visualizations:** {'Available' if HAS_PLOTLY else 'Not Available'}", unsafe_allow_html=True)

# Quick actions in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Actions")
if st.sidebar.button("🔄 Clear Chat History", use_container_width=True):
    st.session_state["messages"] = []
    st.session_state["context"] = deque(maxlen=MAX_CONTEXT)
    st.session_state.in_booking = False
    st.session_state.booking_state = {}
    st.rerun()

if st.sidebar.button("📋 Common Questions", use_container_width=True):
    st.sidebar.info("Frequently asked questions:")
    
    # Show questions from intents
    for intent in intents.get("intents", [])[:5]:  # Show first 5 intents
        if intent.get("patterns"):
            st.sidebar.write(f"• {intent['patterns'][0]}")
    
    # Show questions from FAQ if available
    if faq_df is not None and not faq_df.empty:
        for i, row in faq_df.head(3).iterrows():
            st.sidebar.write(f"• {row['question']}")

# Main content area
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["💬 Chat", "📊 Analytics", "📜 History", "⚙️ Settings", "🏫 University Info", "📊 Evaluation"]
)

# session init
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context" not in st.session_state:
    st.session_state["context"] = deque(maxlen=MAX_CONTEXT)
if "speak_replies" not in st.session_state:
    st.session_state["speak_replies"] = False
if "listening" not in st.session_state:
    st.session_state["listening"] = False
if "input_key" not in st.session_state:
    st.session_state["input_key"] = 0
if "in_booking" not in st.session_state:
    st.session_state.in_booking = False
if "booking_state" not in st.session_state:
    st.session_state.booking_state = {}

# --- Chatbot Tab ---
with tab1:
    st.subheader("💬 Chat with University Assistant")
    
    # Display welcome message if no messages yet
    if not st.session_state["messages"]:
        welcome_msg = f"👋 Welcome to {UNIVERSITY_INFO['name']}! I'm here to help with admissions, programs, campus life, and more. How can I assist you today?"
        st.session_state["messages"].append(("Bot", welcome_msg, "welcome", 1.0, selected_lang_display))
        log_history("Bot", welcome_msg)
    
    # Suggested questions
    st.markdown("**💡 Common questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    suggested_questions = [
        "What are the admission requirements?",
        "How much is tuition?",
        "Schedule a campus tour",
        "What programs do you offer?"
    ]
    
    with col1:
        if st.button(suggested_questions[0], key="suggest1", use_container_width=True):
            process_user_input(suggested_questions[0])
            st.session_state["input_key"] += 1
            st.rerun()
    with col2:
        if st.button(suggested_questions[1], key="suggest2", use_container_width=True):
            process_user_input(suggested_questions[1])
            st.session_state["input_key"] += 1
            st.rerun()
    with col3:
        if st.button(suggested_questions[2], key="suggest3", use_container_width=True):
            process_user_input(suggested_questions[2])
            st.session_state["input_key"] += 1
            st.rerun()
    with col4:
        if st.button(suggested_questions[3], key="suggest4", use_container_width=True):
            process_user_input(suggested_questions[3])
            st.session_state["input_key"] += 1
            st.rerun()
    
    # Context memory display
    if st.session_state["context"]:
        st.markdown(f"""
        <div class="context-memory">
            <strong>🧠 Recent conversation:</strong> {', '.join(list(st.session_state["context"])[-3:])}
        </div>
        """, unsafe_allow_html=True)
    
    # Booking progress indicator
    if st.session_state.get("in_booking", False):
        progress_steps = ["Type", "Date", "Time", "Contact Info"]
        current_step = len(st.session_state.booking_state)
        progress_text = " → ".join([f"**{step}**" if i < current_step else step for i, step in enumerate(progress_steps)])
        st.markdown(f"""
        <div class="context-memory">
            <strong>📋 Scheduling Progress:</strong> {progress_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Input area with columns
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Type your question here...", 
                                  key=f"user_input_{st.session_state['input_key']}",
                                  placeholder="Ask about admissions, programs, campus life...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if HAS_SPEECH:
            if st.button("🎤 Voice Input", key="mic_btn", use_container_width=True):
                st.session_state["listening"] = True
        
        # Additional options
        st.session_state["speak_replies"] = st.checkbox("🔊 Voice replies", value=st.session_state["speak_replies"])
    
    # Handle speech recognition
    if st.session_state.get("listening", False):
        recognized_text, error = recognize_speech()
        if recognized_text:
            user_input = recognized_text
            st.success(f"Recognized: {recognized_text}")
            process_user_input(recognized_text)
            st.session_state["input_key"] += 1
            st.rerun()
        elif error:
            st.error(f"Recognition error: {error}")
        st.session_state["listening"] = False

    if user_input:
        process_user_input(user_input)
        st.session_state["input_key"] += 1
        st.rerun()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for i, (speaker, text, tag, conf, lang) in enumerate(st.session_state["messages"]):
            if speaker == "You":
                # Extract entities for user messages
                entities = extract_entities(text)
                entities_html = ""
                if entities:
                    entities_html = f"<div class='message-meta'>Entities: {', '.join([f'{e[0]} ({e[1]})' for e in entities])}</div>"
                
                st.markdown(f"""
                <div class="user-message fade-in">
                    🧑 <b>You</b>: {text}
                    <div class="message-meta">Language: {lang} • {datetime.now().strftime("%H:%M:%S")}</div>
                    {entities_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message fade-in">
                    🎓 <b>University Assistant</b>: {text}
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

# --- Analytics Tab ---
with tab2:
    st.subheader("📊 Chatbot Analytics")
    
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines="skip")
        except Exception as e:
            st.warning(f"Could not read log file: {e}")
            df = pd.DataFrame()
            
        if not df.empty:
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">%s</div>
                    <div class="metric-label">Total Interactions</div>
                </div>
                """ % len(df), unsafe_allow_html=True)
                
            with col2:
                unique_users = df['user_input'].nunique() if 'user_input' in df.columns else 0
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">%s</div>
                    <div class="metric-label">Unique Users</div>
                </div>
                """ % unique_users, unsafe_allow_html=True)
                
            with col3:
                if "confidence" in df.columns:
                    avg_conf = df['confidence'].astype(float).mean()
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">%.2f%%</div>
                        <div class="metric-label">Avg. Confidence</div>
                    </div>
                    """ % (avg_conf * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">N/A</div>
                        <div class="metric-label">Avg. Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col4:
                if "feedback" in df.columns:
                    positive_feedback = df[df["feedback"].notna() & (df["feedback"].astype(str).str.lower().isin(["yes","1","y","true"]))].shape[0]
                    total_feedback = df[df["feedback"].notna()].shape[0]
                    feedback_rate = positive_feedback / total_feedback if total_feedback > 0 else 0
                    st.metric("Positive Feedback", f"{feedback_rate:.2%}")
                else:
                    st.metric("Positive Feedback", "N/A")
            
            # Create tabs for different analytics views
            eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(["📈 Overview", "🗂️ By Intent", "🌐 Languages", "📶 Confidence"])
            
            with eval_tab1:
                st.markdown("<div class='evaluation-chart'>", unsafe_allow_html=True)
                st.subheader("Daily Interaction Trends")
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
                    df = df.dropna(subset=["timestamp"])  # Remove rows with invalid dates
                    
                    df_daily = df.set_index("timestamp").resample("D").size().reset_index(name="count")
                    
                    if HAS_PLOTLY and not df_daily.empty:
                        fig = px.line(df_daily, x="timestamp", y="count", 
                                     title="Daily Interactions Over Time",
                                     labels={"timestamp": "Date", "count": "Number of Interactions"})
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to matplotlib
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df_daily["timestamp"], df_daily["count"])
                        ax.set_title("Daily Interactions Over Time")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Number of Interactions")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate daily trends: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with eval_tab2:
                st.markdown("<div class='evaluation-chart'>", unsafe_allow_html=True)
                st.subheader("Interactions by Intent")
                if "predicted_tag" in df.columns:
                    tag_counts = df['predicted_tag'].value_counts().reset_index()
                    tag_counts.columns = ['Intent', 'Count']
                    
                    if HAS_PLOTLY:
                        fig = px.pie(tag_counts, values='Count', names='Intent', 
                                    title="Distribution of Interactions by Intent",
                                    height=400)
                        fig.update_layout(showlegend=True, legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.05
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to matplotlib
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.pie(tag_counts['Count'], labels=tag_counts['Intent'], autopct='%1.1f%%')
                        ax.set_title("Distribution of Interactions by Intent")
                        st.pyplot(fig)
                    
                    # Show top intents in a table
                    st.subheader("Top Intents")
                    st.dataframe(tag_counts.head(10), use_container_width=True)
                else:
                    st.info("No intent data available in logs.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with eval_tab3:
                st.markdown("<div class='evaluation-chart'>", unsafe_allow_html=True)
                st.subheader("Language Distribution")
                if "user_lang" in df.columns:
                    lang_counts = df['user_lang'].value_counts().reset_index()
                    lang_counts.columns = ['Language', 'Count']
                    
                    if HAS_PLOTLY:
                        fig = px.bar(lang_counts, x='Language', y='Count', 
                                    title="User Messages by Language",
                                    color='Count', color_continuous_scale='Blues')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback to matplotlib
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(lang_counts["Language"], lang_counts["Count"])
                        ax.set_title("User Messages by Language")
                        ax.set_xlabel("Language")
                        ax.set_ylabel("Count")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                else:
                    st.info("No language data available in logs.")
                st.markdown("</div>", unsafe_allow_html=True)
                    
            with eval_tab4:
                st.markdown("<div class='evaluation-chart'>", unsafe_allow_html=True)
                st.subheader("Confidence Distribution")
                if "confidence" in df.columns:
                    try:
                        conf_df = df[df["confidence"].notna()]
                        conf_df["confidence"] = pd.to_numeric(conf_df["confidence"], errors='coerce')
                        conf_df = conf_df.dropna(subset=["confidence"])
                        
                        if not conf_df.empty:
                            if HAS_PLOTLY:
                                fig = px.histogram(conf_df, x="confidence", 
                                                  title="Distribution of Confidence Scores",
                                                  labels={"confidence": "Confidence Score"},
                                                  nbins=20)
                                fig.update_layout(bargap=0.1, height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Fallback to matplotlib
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.hist(conf_df["confidence"], bins=20)
                                ax.set_title("Distribution of Confidence Scores")
                                ax.set_xlabel("Confidence Score")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                    except Exception as e:
                        st.info(f"Could not generate confidence distribution: {e}")
                
                # Confidence by intent
                if "confidence" in df.columns and "predicted_tag" in df.columns:
                    st.subheader("Confidence by Intent")
                    try:
                        conf_by_intent = df.groupby("predicted_tag")["confidence"].mean().reset_index()
                        conf_by_intent.columns = ['Intent', 'Avg Confidence']
                        conf_by_intent = conf_by_intent.sort_values('Avg Confidence', ascending=False)
                        
                        if HAS_PLOTLY and not conf_by_intent.empty:
                            fig = px.bar(conf_by_intent, x='Intent', y='Avg Confidence',
                                        title="Average Confidence by Intent",
                                        color='Avg Confidence', 
                                        color_continuous_scale='Viridis')
                            fig.update_layout(height=400, xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.bar(conf_by_intent['Intent'], conf_by_intent['Avg Confidence'])
                            ax.set_title("Average Confidence by Intent")
                            ax.set_xlabel("Intent")
                            ax.set_ylabel("Average Confidence")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Could not generate confidence by intent: {e}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Download button
            col_a, col_b = st.columns(2)
            with col_a:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Chat Logs", csv_bytes, "university_chatbot_logs.csv", "text/csv")
        else:
            st.info("No logs yet. Start chatting to generate analytics!")
    else:
        st.info("Log file not found. Start chatting to create one.")

# --- History Tab ---
with tab3:
    st.subheader("📜 Conversation History")
    
    # Add filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_speaker = st.selectbox("Filter by speaker:", ["All", "User", "Bot"])
    with col2:
        date_filter = st.selectbox("Filter by date:", ["All time", "Today", "Last 7 days", "Last 30 days"])
    with col3:
        search_term = st.text_input("Search messages:")
    
    df = pd.read_csv(HISTORY_FILE, on_bad_lines="skip") if os.path.exists(HISTORY_FILE) else pd.DataFrame()
    
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Apply date filter
        if date_filter != "All time":
            now = datetime.now()
            if date_filter == "Today":
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
                df = df[df["timestamp"] >= start_date]
            elif date_filter == "Last 7 days":
                start_date = now - timedelta(days=7)
                df = df[df["timestamp"] >= start_date]
            elif date_filter == "Last 30 days":
                start_date = now - timedelta(days=30)
                df = df[df["timestamp"] >= start_date]
        
        # Apply speaker filter
        if filter_speaker != "All":
            df = df[df['speaker'] == filter_speaker]
            
        # Apply search filter
        if search_term:
            df = df[df['message'].str.contains(search_term, case=False, na=False)]
        
        # Display in a more chat-like format
        for _, row in df.tail(50).iterrows():  # Show only last 50 messages for performance
            timestamp = row['timestamp'].strftime("%H:%M:%S") if 'timestamp' in row else "N/A"
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
                    🎓 <b>University Assistant</b>: {row["message"]}
                    <div class="message-meta">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        st.write(f"Showing {len(df)} of {pd.read_csv(HISTORY_FILE, on_bad_lines='skip').shape[0]} total messages")
        
        col1, col2 = st.columns(2)
        with col1:
            csv_history = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Filtered History", csv_history, "filtered_chat_history.csv", "text/csv", key="download-filtered-history")
        with col2:
            full_history = pd.read_csv(HISTORY_FILE, on_bad_lines="skip").to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full History", full_history, "full_chat_history.csv", "text/csv", key="download-full-history")
    else:
        st.info("No chat history yet. Start a conversation!")

# --- Settings Tab ---
with tab4:
    st.subheader("⚙️ Settings & Feedback")
    
    st.info("Configure the chatbot and provide feedback on your experience.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**System Configuration**")
        st.write(f"Sentence-BERT available: {'✅' if bool(embedder) else '❌'}")
        st.write(f"spaCy loaded: {'✅' if bool(nlp) else '❌'}")
        st.write(f"Language detect available: {'✅' if HAS_LANGDETECT else '❌'}")
        st.write(f"Translation available: {'✅' if (HAS_DEEP_TRANSLATOR or HAS_GOOGLETRANS) else '❌'}")
        st.write(f"Voice I/O: {'✅' if HAS_SPEECH else '❌'}")
        st.write(f"Visualizations: {'✅' if HAS_PLOTLY else '❌'}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Performance Settings**")
        sim_val = st.slider("Semantic similarity threshold", 0.4, 0.9, float(SIM_THRESHOLD), 0.01, key="sim_slider")
        prob_val = st.slider("Probability threshold", 0.5, 0.95, float(PROB_THRESHOLD), 0.01, key="prob_slider")
        context_val = st.slider("Context memory size", 3, 10, MAX_CONTEXT, 1, key="context_slider")
        
        if st.button("Apply Settings", key="apply_btn"):
            SIM_THRESHOLD = sim_val
            PROB_THRESHOLD = prob_val
            st.session_state["context"] = deque(st.session_state["context"], maxlen=context_val)
            st.success(f"Applied new settings!")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Rate Your Experience**")
        st.write("How would you rate your experience with our chatbot?")
        
        rating = st.radio(
            "Select a rating:",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x,
            horizontal=True
        )
        
        feedback_text = st.text_area("Additional feedback (optional):", height=100)
        
        if st.button("Submit Rating", key="rating_btn"):
            ensure_csv(os.path.join(DATA_DIR, "ratings.csv"), ["timestamp", "rating"])
            with open(os.path.join(DATA_DIR, "ratings.csv"), "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rating])
            
            if feedback_text:
                with open(os.path.join(DATA_DIR, "user_feedback.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Rating: {rating}/5\nFeedback: {feedback_text}\n")
            
            st.success(f"Thanks for your feedback! You rated us {rating} ⭐")
            
            if rating <= 2:
                st.info("We're sorry to hear about your experience. Our team will review your feedback.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Export Data**")
        
        if st.button("Export All Chat Data", key="export_btn"):
            # Create a zip file with all data
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                if os.path.exists(LOG_FILE):
                    zip_file.write(LOG_FILE)
                if os.path.exists(HISTORY_FILE):
                    zip_file.write(HISTORY_FILE)
                if os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
                    zip_file.write(os.path.join(DATA_DIR, "ratings.csv"))
                if os.path.exists(os.path.join(DATA_DIR, "user_feedback.txt")):
                    zip_file.write(os.path.join(DATA_DIR, "user_feedback.txt"))
                if os.path.exists(os.path.join(DATA_DIR, "university_bookings.csv")):
                    zip_file.write(os.path.join(DATA_DIR, "university_bookings.csv"))
            
            zip_buffer.seek(0)
            st.download_button(
                label="Download Data Export",
                data=zip_buffer,
                file_name="university_chatbot_data.zip",
                mime="application/zip"
            )
        st.markdown("</div>", unsafe_allow_html=True)

# --- University Info Tab ---
with tab5:
    st.subheader("🏫 University Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Contact Information**")
        st.write(f"**Email:** {UNIVERSITY_INFO['email']}")
        st.write(f"**Phone:** {UNIVERSITY_INFO['phone']}")
        st.write(f"**Address:** {UNIVERSITY_INFO['address']}")
        st.write(f"**Office Hours:** {UNIVERSITY_INFO['hours']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Academic Departments**")
        for department in UNIVERSITY_INFO["departments"]:
            st.write(f"• {department}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Academic Calendar**")
        for semester, dates in UNIVERSITY_INFO["semester_dates"].items():
            st.write(f"**{semester.capitalize()} Semester:** {dates}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.write("**Quick Links**")
        st.write("• [University Website](https://www.university-tech.edu)")
        st.write("• [Admissions Portal](https://apply.university-tech.edu)")
        st.write("• [Course Catalog](https://catalog.university-tech.edu)")
        st.write("• [Campus Map](https://map.university-tech.edu)")
        st.write("• [Student Portal](https://portal.university-tech.edu)")
        st.markdown("</div>", unsafe_allow_html=True)

with tab6:
    st.subheader("📊 Chatbot Evaluation")
    if st.button("Run Evaluation", use_container_width=True):
        evaluate_chatbot(log_file=LOG_FILE, output_file="chatbot_evaluation.csv")

        if os.path.exists("chatbot_evaluation.csv"):
            eval_df = pd.read_csv("chatbot_evaluation.csv")
            st.dataframe(eval_df)

        if os.path.exists("chatbot_evaluation_dashboard.png"):
            st.image("chatbot_evaluation_dashboard.png", caption="Evaluation Dashboard")


st.markdown("---")
st.caption(f"© {datetime.now().year} {UNIVERSITY_INFO['name']}. All rights reserved. | Chatbot version 2.0")
