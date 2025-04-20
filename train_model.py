import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
IMG_SIZE = 224
DATA_DIR = "data_preprocessed"
CATEGORIES = ["COVID", "NORMAL"]

def load_data():
    data = []
    for label, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            data.append([img, label])
    
    if len(data) == 0:
        raise RuntimeError("No data found in the dataset folders.")
    
    np.random.shuffle(data)
    X = np.array([item[0] for item in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array([item[1] for item in data])
    return X, y

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_metrics(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/metrics.png")
    plt.close()

def main(args):
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[INFO] Forced CPU mode.")

    print(f"[INFO] Loading and preprocessing data from {DATA_DIR}...")
    X, y = load_data()
    print(f"[INFO] Total images loaded: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Building and training model...")
    model = build_model()
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1
    )

    os.makedirs("results", exist_ok=True)
    model.save("results/covid_cnn_model.h5")
    plot_metrics(history)

    print("[INFO] Model and training plot saved.")

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    report = classification_report(y_test, y_pred_classes, target_names=CATEGORIES, output_dict=True)

    # Save full report to JSON
    with open("results/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # Print metrics for shell parsing
    for label in CATEGORIES:
        p = report[label]['precision']
        r = report[label]['recall']
        f1 = report[label]['f1-score']
        s = report[label]['support']
        print(f"[METRIC] {label} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | Support: {s}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on COVID X-ray dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU (default: False)")

    args = parser.parse_args()
    main(args)

