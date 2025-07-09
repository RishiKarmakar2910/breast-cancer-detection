import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle

DATA_DIR = 'datasets'
IMG_SIZE = 64  # ✅ Reduced from 128
BATCH_SIZE = 10000
SAVE_DIR = 'preprocessed_batches'

# Optional: limit total number of images (useful for testing)
MAX_IMAGES = None  # Set to an integer like 50000 for testing, or None for full dataset

def save_batch(X, y, batch_index):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'batch_{batch_index}.pkl'), 'wb') as f:
        pickle.dump((X, y), f)
    print(f"✅ Saved batch_{batch_index}.pkl with {len(X)} samples.")

def load_and_save_batches():
    X, y = [], []
    batch_index = 0
    image_counter = 0

    benign_count = 0
    malignant_count = 0

    for patient_folder in os.listdir(DATA_DIR):
        patient_path = os.path.join(DATA_DIR, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        for label in ['0', '1']:
            class_path = os.path.join(patient_path, label)
            if not os.path.isdir(class_path):
                continue

            for img_name in tqdm(os.listdir(class_path), desc=f"Processing {patient_folder}/{label}"):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(int(label))
                    image_counter += 1

                    # Count labels
                    if label == '0':
                        benign_count += 1
                    else:
                        malignant_count += 1

                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")

                if len(X) >= BATCH_SIZE:
                    save_batch(X, y, batch_index)
                    X, y = [], []
                    batch_index += 1

                if MAX_IMAGES and image_counter >= MAX_IMAGES:
                    print(f"⚠️ Reached MAX_IMAGES limit of {MAX_IMAGES}.")
                    if X:
                        save_batch(X, y, batch_index)
                    print(f"\n📊 Final Image Count: Benign = {benign_count}, Malignant = {malignant_count}")
                    return

    if X:
        save_batch(X, y, batch_index)

    # ✅ Final class count summary
    print(f"\n📊 Final Image Count: Benign = {benign_count}, Malignant = {malignant_count}")


if __name__ == "__main__":
    load_and_save_batches()
