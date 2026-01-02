# courier_chatbot_app.py
# Fully functional Courier Tracking Chatbot using NLP, NLTK, BoW, PyTorch, and Streamlit

import streamlit as st
import nltk
import numpy as np
import torch
import torch.nn as nn
import random
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta

# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Intents (Training Data)
# -----------------------------
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
            "responses": ["Sure! Please provide your tracking number."]
        },
        {
            "tag": "delivery_status",
            "patterns": [
                "delivery status",
                "is my parcel delivered",
                "current status of my courier",
                "has my package arrived"
            ],
            "responses": ["Let me check the delivery status. Please share the tracking number."]
        },
        {
            "tag": "expected_delivery",
            "patterns": [
                "expected delivery date",
                "when will my parcel arrive",
                "delivery date",
                "estimated delivery"
            ],
            "responses": ["Please provide the tracking number to check the expected delivery date."]
        },
        {
            "tag": "greeting",
            "patterns": ["hi", "hello", "hey", "good morning"],
            "responses": ["Hello! I can help you track your courier."]
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "exit", "quit"],
            "responses": ["Goodbye! Have a great day."]
        }
    ]
}

# -----------------------------
# NLP Preprocessing
# -----------------------------
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

# -----------------------------
# Training Data (Bag of Words)
# -----------------------------
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

# -----------------------------
# Neural Network Model
# -----------------------------
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = ChatbotModel(len(words), 8, len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Train Model
# -----------------------------
for epoch in range(300):
    outputs = model(X_train)
    loss = criterion(outputs, torch.argmax(y_train, dim=1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------
# Helper Functions
# -----------------------------
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = np.zeros(len(words), dtype=np.float32)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return bag

def predict_class(sentence):
    bow = bag_of_words(sentence)
    bow = torch.from_numpy(bow).float()
    output = model(bow)
    _, predicted = torch.max(output, dim=0)
    return classes[predicted.item()]

# -----------------------------
# Fake Courier Tracking Logic
# -----------------------------
def get_tracking_info(tracking_number):
    status_list = ["In Transit", "Out for Delivery", "Delivered", "Delayed"]
    status = random.choice(status_list)
    expected_date = datetime.now() + timedelta(days=random.randint(1, 5))
    return status, expected_date.strftime("%Y-%m-%d")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📦 Courier Tracking Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    intent = predict_class(user_input)
    response = ""

    if intent in ["track_parcel", "delivery_status", "expected_delivery"]:
        response = "Please enter your tracking number (e.g. TRK123456)."
    else:
        for i in intents['intents']:
            if i['tag'] == intent:
                response = random.choice(i['responses'])

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

tracking_number = st.text_input("Tracking Number")

if st.button("Track Parcel") and tracking_number:
    status, date = get_tracking_info(tracking_number)
    st.session_state.chat.append(("Bot", f"📦 Status: {status}"))
    st.session_state.chat.append(("Bot", f"📅 Expected Delivery Date: {date}"))

st.write("---")
for speaker, msg in st.session_state.chat:
    st.write(f"**{speaker}:** {msg}")

# -----------------------------
# Run with: streamlit run courier_chatbot_app.py
# -----------------------------
