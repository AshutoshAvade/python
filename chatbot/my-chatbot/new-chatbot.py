import streamlit as st
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------
# NLTK Downloads
# ---------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="ML Courier Tracking Chatbot",
    page_icon="📦",
    layout="centered"
)

st.title("📦 ML-Based Courier Tracking Chatbot")
st.caption("Track parcels using ML & NLP")

# ---------------------------------
# Load Courier Data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("courier_tracking_data.csv")
    df["tracking_number"] = df["tracking_number"].astype(str).str.upper().str.strip()
    return df

df = load_data()

# ---------------------------------
# NLP Preprocessing
# ---------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ---------------------------------
# Intent Classification Model
# ---------------------------------
training_sentences = [
    "track my parcel", "where is my shipment", "track courier",
    "what is the status", "delivery status", "parcel status",
    "shipping cost", "what is the price", "delivery charges",
    "which courier", "carrier name", "shipping company",
    "customer name", "who is the recipient", "receiver name",
    "hi", "hello", "hey", "good morning"
]

training_labels = [
    "track", "track", "track",
    "status", "status", "status",
    "price", "price", "price",
    "courier", "courier", "courier",
    "customer", "customer", "customer",
    "greeting", "greeting", "greeting", "greeting"
]

vectorizer = TfidfVectorizer(preprocessor=preprocess)
X_train = vectorizer.fit_transform(training_sentences)

intent_model = LogisticRegression()
intent_model.fit(X_train, training_labels)

def predict_intent(text):
    return intent_model.predict(vectorizer.transform([text]))[0]

# ---------------------------------
# Entity Extraction
# ---------------------------------
def extract_tracking_number(text):
    match = re.search(r"([A-Z]{2,4})(\d+)", text.upper())
    return match.group(1) + match.group(2) if match else None

def find_by_tracking(tracking):
    for _, row in df.iterrows():
        if row["tracking_number"].startswith(tracking):
            return row
    return None

# ---------------------------------
# Small Talk / Conversation Rules
# ---------------------------------
def conversation_reply(text):
    text = text.lower().strip()

    if text in ["hi", "hello", "hey"]:
        return "Hello 👋 Please provide your tracking number."

    if text in ["thanks", "thank you", "thankyou"]:
        return "You're welcome 😊 Let me know if you need anything else."

    if text in ["bye", "goodbye", "exit"]:
        return "Goodbye 👋 Have a great day!"

    return None

# ---------------------------------
# Response Generator
# ---------------------------------
def generate_response(row, intent):
    customer_name = f"{row['Customer Fname']} {row['Customer Lname']}"

    if intent == "customer":
        return f"👤 **Customer name for tracking number {row['tracking_number']} is {customer_name}**"

    if intent == "status":
        return f"📦 **Status of tracking number {row['tracking_number']} is {row['status']}**"

    if intent == "price":
        return f"💰 **Cost for tracking number {row['tracking_number']} is ₹{row['Cost']}**"

    if intent == "courier":
        return (
            f"🚚 **Carrier:** {row['Carrier']}\n\n"
            f"📍 **Route:** {row['Origin_Warehouse']} → {row['Destination']}"
        )

    return (
        f"📦 **Status:** {row['status']}\n\n"
        f"👤 **Customer:** {customer_name}\n\n"
        f"🚚 **Carrier:** {row['Carrier']}\n\n"
        f"📍 **Route:** {row['Origin_Warehouse']} → {row['Destination']}\n\n"
        f"💰 **Cost:** ₹{row['Cost']}\n\n"
        f"📅 **Shipment Date:** {row['Shipment_Date']}\n\n"
        f"📅 **Expected Delivery:** {row['expected_delivery_date']}\n\n"
        f"📦 **Weight:** {row['Weight_kg']} kg\n\n"
        f"🚛 **Transit Days:** {row['Transit_Days']}"
    )

# ---------------------------------
# Chat History
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------
# Chat Input
# ---------------------------------
user_input = st.chat_input("Ask about your parcel...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- RULE-BASED SMALL TALK FIRST ----
    bot_reply = conversation_reply(user_input)

    if not bot_reply:
        tracking = extract_tracking_number(user_input)
        text_lower = user_input.lower()

        if tracking:
            if "status" in text_lower:
                intent = "status"
            elif "cost" in text_lower or "price" in text_lower:
                intent = "price"
            elif "courier" in text_lower or "carrier" in text_lower:
                intent = "courier"
            elif "customer" in text_lower or "name" in text_lower or "recipient" in text_lower:
                intent = "customer"
            else:
                intent = "track"
        else:
            intent = predict_intent(user_input)

        if intent == "greeting":
            bot_reply = "Hello 👋 Please provide your tracking number."
        else:
            row = find_by_tracking(tracking) if tracking else None
            bot_reply = generate_response(row, intent) if row is not None else "❌ Sorry, I couldn't find any parcel details."

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
