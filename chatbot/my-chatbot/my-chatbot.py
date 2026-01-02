# courier_chatbot_bert_chatflow_app.py
# Single-file Courier Tracking Chatbot
# Fine-tuning BERT (optional) + Inference + Rule-based Tracking + Exit Intent

import streamlit as st
import pandas as pd
import torch
import re
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# =============================
# CONFIG
# =============================
MODEL_DIR = "bert_intent_model"
EXIT_WORDS = ["bye", "exit", "quit", "end", "close"]

# =============================
# LOAD TRACKING DATA
# =============================
@st.cache_data
def load_tracking_data():
    return pd.read_csv("courier_tracking_data.csv")

df = load_tracking_data()

# =============================
# INTENT DATA (FOR FINE-TUNING)
# =============================
intent_data = pd.DataFrame({
    "text": [
        "hello", "hi", "hey", "good morning",
        "bye", "exit", "quit",
        "track my parcel", "track my courier",
        "where is my package", "delivery status",
        "when will it arrive", "expected delivery"
    ],
    "label": [
        "greeting", "greeting", "greeting", "greeting",
        "goodbye", "goodbye", "goodbye",
        "track_parcel", "track_parcel",
        "delivery_status", "delivery_status",
        "expected_delivery", "expected_delivery"
    ]
})

# =============================
# TRAIN MODEL (RUNS ONCE)
# =============================
def train_model():
    le = LabelEncoder()
    intent_data["label"] = le.fit_transform(intent_data["label"])

    dataset = Dataset.from_pandas(intent_data)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(le.classes_),
        id2label=dict(enumerate(le.classes_)),
        label2id={v: k for k, v in enumerate(le.classes_)}
    )

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        logging_steps=5,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

# Train only if model not exists
if not os.path.exists(MODEL_DIR):
    train_model()

# =============================
# LOAD MODEL (INFERENCE)
# =============================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# =============================
# HELPERS
# =============================
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return model.config.id2label[torch.argmax(outputs.logits).item()]


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
st.caption("Chat naturally – enter tracking number anytime")

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
    user_text = user_input.lower().strip()

    if user_text in EXIT_WORDS:
        bot_reply = "Goodbye! Have a great day 👋"
        st.session_state.awaiting_tracking = False

    elif is_tracking_number(user_input):
        result = get_tracking_info(user_input)
        if result:
            status, date = result
            bot_reply = f"📦 **Status:** {status}\n\n📅 **Expected Delivery:** {date}"
        else:
            bot_reply = "❌ Tracking number not found."
        st.session_state.awaiting_tracking = False

    elif st.session_state.awaiting_tracking:
        bot_reply = "❌ Tracking number not found."
        st.session_state.awaiting_tracking = False

    else:
        intent = predict_intent(user_input)
        if intent in ["track_parcel", "delivery_status", "expected_delivery"]:
            bot_reply = "Sure 🙂 Please enter your tracking number."
            st.session_state.awaiting_tracking = True
        elif intent == "greeting":
            bot_reply = "Hello! You can ask me to track your courier anytime."
        elif intent == "goodbye":
            bot_reply = "Goodbye! Have a great day 👋"
        else:
            bot_reply = "Sorry, I didn’t understand that."

    st.session_state.messages.append(("assistant", bot_reply))
    st.rerun()

# =============================
# RUN:
# streamlit run courier_chatbot_bert_chatflow_app.py
# =============================
