import os
import cv2
import numpy as np
from tqdm import tqdm

DATA_DIR = 'data'
OUTPUT_DIR = 'data_preprocessed'
IMG_SIZE = 224
CATEGORIES = ['COVID', 'Normal']

def preprocess_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for category in CATEGORIES:
        input_folder = os.path.join(DATA_DIR, category)
        output_folder = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in tqdm(os.listdir(input_folder), desc=f"Processing {category}"):
            try:
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(input_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[WARNING] Skipping corrupt image: {img_path}")
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize

                out_path = os.path.join(output_folder, img_name)
                cv2.imwrite(out_path, (img * 255).astype(np.uint8))

            except Exception as e:
                print(f"[ERROR] Could not process {img_name}: {e}")

if __name__ == "__main__":
    preprocess_and_save()

