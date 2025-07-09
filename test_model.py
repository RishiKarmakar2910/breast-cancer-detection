import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# === CONFIG ===
MODEL_PATH = "models/breast_cancer_detector_final.h5"  # ✅ use the latest model
IMG_SIZE = 64  # Match your training size

# === Load the Model ===
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# === Load and Preprocess a Test Image ===
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)    # Shape: (1, 64, 64, 3)
    return img

# === Predict on a Sample Image ===
def predict_image(img_path):
    image = preprocess_image(img_path)
    prediction = model.predict(image)[0][0]

    label = "Malignant" if prediction > 0.5 else "Benign"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"🩺 Prediction: {label} ({confidence * 100:.2f}% confidence)")

    # Show the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"{label} ({confidence * 100:.2f}% confidence)")
    plt.axis("off")
    plt.show()

# === Example Usage ===
if __name__ == "__main__":
    test_image_path = "sample_test_image.png"  #  Replace with a real image path
    predict_image(test_image_path)
    test_image_path2 = "sample_test_image2.png"
    predict_image(test_image_path2)
