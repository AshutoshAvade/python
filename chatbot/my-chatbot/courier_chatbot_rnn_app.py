# courier_chatbot_rnn_app.py
# Courier Tracking Chatbot using NLP + TF-IDF + RNN (LSTM)
# Includes Greeting, Tracking, Status, Expected Delivery, Thank You, Goodbye

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# =============================
# CONFIG
# =============================
MODEL_PATH = "rnn_intent_model.pth"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# =============================
# LOAD TRACKING DATA
# =============================
@st.cache_data
def load_tracking_data():
    return pd.read_csv("courier_tracking_data.csv")

df = load_tracking_data()

# =============================
# INTENT DATA
# =============================
intent_data = pd.DataFrame({
    "text": [
        "hi", "hello", "hey",
        "track my parcel", "where is my courier",
        "delivery status", "expected delivery",
        "thanks", "thank you", "thanks a lot",
        "bye", "exit", "quit"
    ],
    "label": [
        "greeting", "greeting", "greeting",
        "track_parcel", "delivery_status",
        "delivery_status", "expected_delivery",
        "thank_you", "thankyou", "thank-you",
        "goodbye", "goodbye", "goodbye"
    ]
})

# =============================
# RNN MODEL
# =============================
class IntentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# =============================
# TRAIN MODEL (ONCE)
# =============================
def train_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(intent_data['text']).toarray()

    le = LabelEncoder()
    y = le.fit_transform(intent_data['label'])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = IntentRNN(X.shape[1], 64, len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    torch.save({
        "model_state": model.state_dict(),
        "labels": le.classes_
    }, MODEL_PATH)

    import pickle
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

# Train if needed
if not os.path.exists(MODEL_PATH):
    train_model()

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    # PyTorch 2.6+ fix: explicitly allow full loading from trusted local file
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    labels = checkpoint['labels']

    import pickle
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    model = IntentRNN(len(vectorizer.get_feature_names_out()), 64, len(labels))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, vectorizer, labels

model, vectorizer, labels = load_model()

# =============================
# HELPERS
# =============================
def predict_intent(text):
    vec = vectorizer.transform([text]).toarray()
    with torch.no_grad():
        output = model(torch.tensor(vec, dtype=torch.float32))
    return labels[torch.argmax(output).item()]


def is_tracking_number(text):
    return bool(re.match(r"^[A-Z]{2,3}[0-9]{4,10}$", text.strip().upper()))


def get_tracking_info(tracking_number):
    record = df[df['tracking_number'].astype(str) == tracking_number]
    if record.empty:
        return None
    row = record.iloc[0]
    return row['status'], row['expected_delivery_date']

# =============================
# STREAMLIT UI
# =============================
st.set_page_config(page_title="Courier Chatbot", layout="centered")
st.title("📦 Courier Tracking Chatbot")
st.caption("NLP + TF-IDF + RNN (LSTM)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_tracking" not in st.session_state:
    st.session_state.awaiting_tracking = False

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(("user", user_input))

    # 1️⃣ Tracking number has highest priority
    if is_tracking_number(user_input):
        result = get_tracking_info(user_input.strip())
        if result:
            status, date = result
            bot_reply = f"📦 **Status:** {status}\n\n📅 **Expected Delivery:** {date}"
        else:
            bot_reply = "❌ Tracking number not found."
        st.session_state.awaiting_tracking = False

    # 2️⃣ NLP Intent Detection
    else:
        intent = predict_intent(user_input.lower())

        if intent == "greeting":
            bot_reply = "Hello! 👋 I can help you track your courier."

        elif intent in ["track_parcel", "delivery_status", "expected_delivery"]:
            bot_reply = "Sure 🙂 Please enter your tracking number."
            st.session_state.awaiting_tracking = True

        elif intent == "thank_you":
            bot_reply = "You're welcome 😊 Happy to help!"

        elif intent == "goodbye":
            bot_reply = "Goodbye! Have a great day 👋"

        else:
            bot_reply = "Sorry, I didn’t understand that."

    st.session_state.messages.append(("assistant", bot_reply))
    st.rerun()

# =============================
# RUN:
# streamlit run courier_chatbot_rnn_app.py
# =============================
