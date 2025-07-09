import streamlit as st

# Page setup
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

# --- Header ---
st.title("🔬 Breast Cancer Detection System")
st.markdown("---")

# --- Description ---
st.markdown("""
### What is Breast Cancer?

Breast cancer is a disease in which cells in the breast grow out of control. There are two main types:

- **Benign Tumors**: Non-cancerous and usually not life-threatening.
- **Malignant Tumors**: Cancerous and can spread to other parts of the body.

### What This App Does

This application uses computer vision and deep learning to analyze histopathology images of breast tissue and classify them into **benign** or **malignant**.

The model was trained on the [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis) and uses convolutional neural networks (CNNs) to detect microscopic patterns.
""")

# --- Buttons for Navigation ---
st.markdown("### 🔍 Choose an Operation Mode")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧪 Prediction (Pre-trained)"):
        st.session_state.model_choice = "prediction"
        st.switch_page("pages/Prediction.py")  # Navigates to Prediction

with col2:
    if st.button("🧫 Detection (Custom-trained)"):
        st.session_state.model_choice = "detection"
        st.switch_page("pages/Detection.py")  # Navigates to Detection
