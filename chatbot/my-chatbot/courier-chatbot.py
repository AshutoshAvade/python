# courier_chatbot_app.py
# Courier Tracking Chatbot using NLP, NLTK, BoW, PyTorch, Streamlit
# Uses a Kaggle-style CSV dataset as a database

import streamlit as st
import nltk
import numpy as np
import torch
import torch.nn as nn
import random
import pandas as pd
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# =============================
# NLTK Setup
# =============================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# =============================
# LOAD KAGGLE DATASET (CSV)
# =============================
# Expected CSV columns:
# tracking_number, status, expected_delivery_date

@st.cache_data
def load_data():
    return pd.read_csv("courier_tracking_data.csv")

df = load_data()

# =============================
# INTENTS DATA
# =============================
intents = {
    "intents": [
        {
            "tag": "track_parcel",
            "patterns": [
                "track my parcel",
                "where is my courier",
                "track shipment",
                "find my package",
                "track order"
            ],
            "responses": ["Please enter your tracking number."]
        },
        {
            "tag": "delivery_status",
            "patterns": [
                "delivery status",
                "is my parcel delivered",
                "current status of my courier"
            ],
            "responses": ["Checking delivery status. Enter tracking number."]
        },
        {
            "tag": "expected_delivery",
            "patterns": [
                "expected delivery date",
                "when will my parcel arrive",
                "estimated delivery"
            ],
            "responses": ["Please provide tracking number for delivery date."]
        },
        {
            "tag": "greeting",
            "patterns": ["hi", "hello", "hey"],
            "responses": ["Hello! I can help track your courier."]
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "exit"],
            "responses": ["Goodbye! Stay safe."]
        }
    ]
}

# =============================
# NLP PREPROCESSING
# =============================
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# =============================
# TRAINING DATA (BAG OF WORDS)
# =============================
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
X_train = torch.tensor(list(training[:, 0]), dtype=torch.float32)
y_train = torch.tensor(list(training[:, 1]), dtype=torch.float32)

# =============================
# NEURAL NETWORK MODEL
# =============================
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = ChatbotModel(len(words), 8, len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    outputs = model(X_train)
    loss = criterion(outputs, torch.argmax(y_train, dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =============================
# HELPER FUNCTIONS
# =============================
def clean_sentence(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

def predict_intent(sentence):
    bow = torch.from_numpy(bag_of_words(sentence))
    output = model(bow)
    _, predicted = torch.max(output, dim=0)
    return classes[predicted.item()]

# =============================
# DATASET-BASED TRACKING LOGIC
# =============================
def get_tracking_info(tracking_number):
    record = df[df['tracking_number'] == tracking_number]
    if record.empty:
        return None
    return record.iloc[0]['status'], record.iloc[0]['expected_delivery_date']

# =============================
# STREAMLIT UI
# =============================
st.title("📦 Courier Tracking Chatbot (Kaggle Dataset)")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    intent = predict_intent(user_input)
    reply = ""
    for i in intents['intents']:
        if i['tag'] == intent:
            reply = random.choice(i['responses'])
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", reply))

tracking_number = st.text_input("Enter Tracking Number")

if st.button("Track") and tracking_number:
    result = get_tracking_info(tracking_number)
    if result:
        status, date = result
        st.session_state.chat.append(("Bot", f"📦 Status: {status}"))
        st.session_state.chat.append(("Bot", f"📅 Expected Delivery Date: {date}"))
    else:
        st.session_state.chat.append(("Bot", "❌ Tracking number not found."))

st.write("---")
for speaker, msg in st.session_state.chat:
    st.write(f"**{speaker}:** {msg}")

# =============================
# RUN: streamlit run courier_chatbot_app.py
# =============================
