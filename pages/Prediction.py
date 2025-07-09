import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Set page config
st.set_page_config(page_title="Predict Tumor Type", layout="centered")

# --- Header ---
st.title("🩺 Predict Breast Tumor Type")
st.markdown("---")

# --- Image Preprocessing ---
IMG_SIZE = 64  # Use 64 if that's what your model was trained on

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return img

# --- Load Model Based on Session Choice ---
model_choice = st.session_state.get("model_choice", "prediction")

if model_choice == "prediction":
    model_path = "breast_cancer_detector_final.h5"
else:
    model_path = "model_after_batch_4.h5"  # or latest detection model

try:
    model = load_model(model_path)
    st.success(f"✅ Loaded **{model_choice}** model successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# --- Image Upload ---
uploaded_file = st.file_uploader("📤 Upload a histopathology image for prediction", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        processed = preprocess_image(image)
        prediction = float(model.predict(processed)[0][0])

        # Threshold logic with confidence band
        if prediction < 0.4:
            confidence = (1 - prediction) * 100
            st.success(f"🟢 **Benign Tumor** detected with `{confidence:.2f}%` confidence.")
        elif prediction > 0.6:
            confidence = prediction * 100
            st.error(f"🔴 **Malignant Tumor** detected with `{confidence:.2f}%` confidence.")
        else:
            confidence = abs(0.5 - prediction) * 200
            st.warning(f"⚠️ **Uncertain** prediction. Confidence margin: `{confidence:.2f}%`. Consider rechecking or consulting a doctor.")
else:
    st.info("👆 Upload an image to begin prediction.")
