import streamlit as st
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(page_title="Smart Dustbin", layout="centered")
st.title("🗑️ Smart Dustbin Overflow Detection")
st.write("Deep Learning + NLP | Automatic Email Alert System")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# EMAIL CONFIG
# ---------------------------------
SENDER_EMAIL = "tatyavinchu057@gmail.com"
RECEIVER_EMAIL = "ashuavade0@gmail.com"
GMAIL_APP_PASSWORD = "oqgkhhcamgkrrjoe"

def send_email_alert():
    try:
        msg = MIMEText(
            "🚨 ALERT: Dustbin is OVERFLOWING.\n\nImmediate cleaning required."
        )
        msg["Subject"] = "Smart Dustbin Overflow Alert"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)

        st.success("📧 Email alert sent successfully")

    except Exception as e:
        st.error(f"❌ Email failed: {e}")

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load("model.pth", map_location=device)

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )
    model.classifier[1] = nn.Linear(1280, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint["classes"]

model, class_names = load_model()

# ---------------------------------
# IMAGE TRANSFORM
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def predict(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, 1)
        conf, pred = torch.max(prob, 1)

    label = class_names[pred.item()].lower()
    return label, conf.item()

# ---------------------------------
# SESSION STATE (ANTI EMAIL SPAM)
# ---------------------------------
if "alert_sent" not in st.session_state:
    st.session_state.alert_sent = False

# ---------------------------------
# IMAGE UPLOAD UI
# ---------------------------------
st.subheader("📸 Upload Dustbin Image")
file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

CONF_THRESHOLD = 0.6

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)

    label, conf = predict(img)

    st.markdown("### 🔍 Detection Result")
    st.info(f"**Status:** {label.capitalize()}")
    st.info(f"**Confidence:** {conf * 100:.2f}%")

    if label == "overflow":
        if conf >= CONF_THRESHOLD:
            st.error("🚨 Dustbin is OVERFLOWING – Immediate cleaning required")

            if not st.session_state.alert_sent:
                send_email_alert()
                st.session_state.alert_sent = True
        else:
            st.warning("⚠️ Possible overflow detected (low confidence)")
            st.session_state.alert_sent = False

    elif label == "half":
        st.warning("⚠️ Dustbin is half filled")
        st.session_state.alert_sent = False

    else:
        st.success("✅ Dustbin is clean")
        st.session_state.alert_sent = False

# ---------------------------------
# NLP COMPLAINT ANALYZER
# ---------------------------------
st.markdown("---")
st.subheader("📝 Citizen Complaint Analyzer")

@st.cache_resource
def load_nlp():
    texts = [
        "bin overflowing", "dustbin full", "bad smell",
        "garbage not collected", "trash everywhere",
        "area dirty", "bin needs cleaning",
        "bin is clean", "area clean", "no issues",
        "good job", "thank you"
    ]
    labels = [1,1,1,1,1,1,1,0,0,0,0,0]

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression()
    clf.fit(X, labels)

    return vectorizer, clf

vectorizer, clf = load_nlp()

complaint = st.text_area("Enter complaint")

if st.button("Analyze Complaint"):
    X_test = vectorizer.transform([complaint.lower()])
    result = clf.predict(X_test)

    if result[0] == 1:
        st.error("🚨 Urgent Complaint Detected")
    else:
        st.success("✅ Normal Complaint")

# ---------------------------------
st.caption("Smart City Project | PyTorch + Streamlit | Automated Monitoring System")
