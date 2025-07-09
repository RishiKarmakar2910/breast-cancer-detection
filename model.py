import os
import pickle
import numpy as np
import gc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# === CONFIG ===
BATCH_DIR = 'preprocessed_batches'
IMG_SIZE = 64        # Reduced from 128
EPOCHS = 2           # Keep low for testing
BATCH_SIZE = 16      # Reduced from 32
USE_BATCHES = 5      # Use only first 5 .pkl files for faster runs

# === Model Architecture ===
def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Load a Preprocessed Batch from .pkl ===
def load_batch(file_path):
    with open(file_path, 'rb') as f:
        X, y = pickle.load(f)
    X = np.array(X, dtype='float32') / 255.0  # Normalize pixel values
    y = np.array(y)
    return X, y

# === Training Loop ===
if __name__ == "__main__":
    model = build_model()
    batch_files = sorted(os.listdir(BATCH_DIR))[:USE_BATCHES]  # Use subset only

    for epoch in range(EPOCHS):
        print(f"\n📘 Epoch {epoch + 1}/{EPOCHS}")

        for i, batch_file in enumerate(batch_files):
            batch_path = os.path.join(BATCH_DIR, batch_file)
            print(f"\n📦 Training on batch: {batch_file}")
            
            X, y = load_batch(batch_path)

            # Split into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=42
            )

            model.fit(
                X_train, y_train,
                epochs=1,
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                verbose=1
            )

            # OPTIONAL: Save intermediate model state
            model.save(f"model_after_batch_{i}.h5")

            # ✅ Free memory
            del X, y, X_train, X_val, y_train, y_val
            gc.collect()
            K.clear_session()

            # Rebuild model and load weights
            model = build_model()
            model.load_weights(f"model_after_batch_{i}.h5")

    # Final model save
    model.save("breast_cancer_detector_final.h5")
    print("\n✅ Final model saved as 'breast_cancer_detector_final.h5'")
