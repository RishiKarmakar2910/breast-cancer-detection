import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Detection Mode", layout="centered")
st.title("🔍 Detection Mode")
st.markdown("---")

# --- Check session state ---
if "model_choice" not in st.session_state:
    st.warning("⚠️ Please go to the Home page and select a model first.")
    st.stop()

model_choice = st.session_state["model_choice"]

# --- Load model based on user selection ---
@st.cache_resource
def load_selected_model():
    if model_choice == "prediction":
        return load_model("breast_cancer_detector_final.h5")
    else:
        return load_model("models/model_after_batch_4.h5")  # Update path as needed

model = load_selected_model()
st.success(f"✅ Loaded **{model_choice}** model for detection")

# --- Image preprocessing ---
IMG_SIZE = 64  # Change if you retrained with a different size

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- File Uploader ---
uploaded_file = st.file_uploader("📤 Upload a histopathology image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔎 Detect"):
        with st.spinner("Analyzing..."):
            img_array = preprocess_image(image)
            prediction = float(model.predict(img_array)[0][0])

        st.markdown("---")
        # Confidence interpretation
        if prediction < 0.4:
            confidence = (1 - prediction) * 100
            st.success(f"🟢 **Benign Tumor** detected with `{confidence:.2f}%` confidence.")
        elif prediction > 0.6:
            confidence = prediction * 100
            st.error(f"🔴 **Malignant Tumor** detected with `{confidence:.2f}%` confidence.")
        else:
            confidence = abs(0.5 - prediction) * 200
            st.warning(f"⚠️ **Uncertain** detection. Confidence margin: `{confidence:.2f}%`. Consider retesting or clinical diagnosis.")
else:
    st.info("👆 Upload a histopathology image to begin detection.")
