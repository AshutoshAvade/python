import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="UPI Fraud Detection", layout="centered")

st.title("💳 UPI Fraud Detection System")
st.markdown("**Hybrid AI – Random Forest + Anomaly Detection (2025)**")

# =========================================================
@st.cache_data
def load_and_train():
    df = pd.read_csv("upi_transactions.csv")

    # -------- CLEAN DATA ----------
    df["TransactionFrequency"] = (
        df["TransactionFrequency"]
        .astype(str)
        .str.extract("(\d+)")
        .astype(float)
    )

    for col in ["UnusualLocation","UnusualAmount","NewDevice"]:
        df[col] = df[col].astype(str).map({
            "1":1,"0":0,"Yes":1,"No":0,"True":1,"False":0
        }).fillna(0)

    df["FailedAttempts"] = pd.to_numeric(df["FailedAttempts"], errors="coerce").fillna(0)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["AvgTransactionAmount"] = pd.to_numeric(df["AvgTransactionAmount"], errors="coerce")

    df = df.dropna()

    # -------- FEATURE ENGINEERING ----------
    df["amount_ratio"] = df["Amount"] / df["AvgTransactionAmount"]

    features = [
        "amount_ratio",
        "TransactionFrequency",
        "UnusualLocation",
        "UnusualAmount",
        "NewDevice",
        "FailedAttempts"
    ]

    X = df[features]
    y = df["FraudFlag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🚀 Fraud-focused Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight={0:1, 1:6},
        max_depth=12,
        min_samples_leaf=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(X_train)

    return rf, iso, scaler, X_test, y_test

# =========================================================
rf, iso, scaler, X_test, y_test = load_and_train()
st.success("Model trained successfully")

# =========================================================
st.subheader("📊 Model Performance")

# Probability-based fraud prediction
rf_probs = rf.predict_proba(X_test)[:,1]
rf_preds = (rf_probs > 0.35).astype(int)

cm = confusion_matrix(y_test, rf_preds)

fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# -------- Classification Report --------
report = classification_report(y_test, rf_preds, output_dict=True)

st.write("Accuracy:", report["accuracy"])

# Fraud label = 1
if 1 in report:
    st.write("Fraud Recall:", report[1]["recall"])
else:
    st.warning("⚠ Fraud recall unavailable — increase fraud data if needed")

st.divider()

# =========================================================
st.subheader("🔍 Check Live Transaction")

Amount = st.number_input("Transaction Amount (₹)", 1.0, 500000.0)
AvgAmount = st.number_input("User Avg Amount (₹)", 1.0, 500000.0)
Frequency = st.slider("Transaction Frequency (per day)", 0, 50)
UnusualLoc = st.selectbox("Unusual Location", [0,1])
UnusualAmt = st.selectbox("Unusual Amount", [0,1])
NewDev = st.selectbox("New Device", [0,1])
Failed = st.slider("Failed Login Attempts", 0, 10)

amount_ratio = Amount / AvgAmount

if st.button("Check Fraud Risk"):
    X = np.array([[amount_ratio, Frequency, UnusualLoc, UnusualAmt, NewDev, Failed]])
    X = scaler.transform(X)

    rf_prob = rf.predict_proba(X)[0][1]
    anomaly = -iso.score_samples(X)[0]

    final_risk = 0.7 * rf_prob + 0.3 * anomaly

    st.subheader("Fraud Probability")
    st.metric("Risk Score", f"{final_risk:.2f}")

    if final_risk > 0.45:
        st.error("🚨 Fraud Detected – Transaction Blocked")
    elif final_risk > 0.30:
        st.warning("⚠ Suspicious – MFA Required")
    else:
        st.success("✅ Transaction Safe")

st.caption("UPI Fraud Detection | AI Powered | 2025")
