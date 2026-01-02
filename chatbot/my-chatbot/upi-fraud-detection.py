import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="UPI Fraud Detection", layout="centered")

st.title("💳 UPI Fraud Detection System")
st.markdown("**Hybrid AI Model – Random Forest + Isolation Forest**")

@st.cache_data
def load_and_train():
    df = pd.read_csv("upi_transactions.csv")

    # Feature engineering
    df["amount_ratio"] = df["amount"] / df["user_avg_amount"]
    df["night_tx"] = df["hour"].apply(lambda x: 1 if x < 5 else 0)

    X = df[["amount_ratio","tx_per_hour","geo_distance","time_delta","night_tx"]]
    y = df["is_fraud"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
    rf.fit(X_scaled, y)

    iso = IsolationForest(contamination=0.03)
    iso.fit(X_scaled)

    return rf, iso, scaler

rf, iso, scaler = load_and_train()

st.success("Models trained successfully")

st.divider()
st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount (₹)", 1.0, 100000.0)
avg_amount = st.number_input("User Average Amount (₹)", 1.0, 100000.0)
tx_per_hour = st.slider("Transactions in last hour", 0, 50)
geo_distance = st.slider("Distance from last location (km)", 0, 2000)
time_delta = st.slider("Seconds since last transaction", 1, 50000)
hour = st.slider("Transaction hour", 0, 23)

amount_ratio = amount / avg_amount
night_tx = 1 if hour < 5 else 0

if st.button("Check Fraud Risk"):
    X = np.array([[amount_ratio, tx_per_hour, geo_distance, time_delta, night_tx]])
    X_scaled = scaler.transform(X)

    rf_prob = rf.predict_proba(X_scaled)[0][1]
    anomaly = -iso.score_samples(X_scaled)[0]

    final_risk = 0.7 * rf_prob + 0.3 * anomaly

    st.subheader("Fraud Risk Score")
    st.metric("Risk", f"{final_risk:.2f}")

    if final_risk > 0.75:
        st.error("🚨 Fraud Detected — Transaction Blocked")
    elif final_risk > 0.5:
        st.warning("⚠ Suspicious — MFA Required")
    else:
        st.success("✅ Transaction Safe")

st.divider()
st.caption("UPI Fraud Detection using AI | 2025")
