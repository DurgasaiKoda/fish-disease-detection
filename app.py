import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import os
import json

# Constants
DATASET_PATH = 'dataset'
MODEL_DIR = 'models'
ACCURACY_FILE = 'accuracies.json'
IMAGE_SIZE = (64, 64)
MAX_IMAGES_PER_CLASS = 100

# Page setup
st.set_page_config(page_title="Fish Disease Classifier", layout="wide")
st.title("🐟 Fish Disease Classification App")

# Create model folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to load image dataset
def load_data_from_directory(base_path, categories, image_size=(64, 64), max_images=100):
    data, labels = [], []
    for label_index, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        count = 0
        for img_file in os.listdir(category_path):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if count >= max_images:
                break
            try:
                img_path = os.path.join(category_path, img_file)
                img = imread(img_path)
                img_resized = resize(img, image_size)
                data.append(img_resized.flatten())
                labels.append(label_index)
                count += 1
            except:
                pass
    return np.array(data), np.array(labels)

# Model definitions
def get_model_definitions():
    return {
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True)),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss')
    }

# Save accuracy scores to JSON
def save_accuracies_to_json(accuracies, filename=ACCURACY_FILE):
    with open(filename, "w") as f:
        json.dump(accuracies, f)

# Load accuracy scores from JSON
def load_accuracies_from_json(filename=ACCURACY_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        return {}

# Train models and save to disk
def train_and_save_models(x_train, y_train, x_valid, y_valid, categories):
    model_defs = get_model_definitions()
    accuracies = {}
    trained_models = {}
    st.subheader("🚀 Training models...")

    for name, model in model_defs.items():
        with st.spinner(f"Training {name}..."):
            model.fit(x_train, y_train)
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
            y_pred = model.predict(x_valid)
            acc = accuracy_score(y_valid, y_pred)
            accuracies[name] = round(acc, 4)
            trained_models[name] = model
            st.success(f"{name} trained with accuracy: {acc * 100:.2f}%")

    save_accuracies_to_json(accuracies)
    return trained_models, accuracies

# Load saved models from disk
def load_saved_models():
    models = {}
    for name in get_model_definitions().keys():
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

# Load category names
categories = os.listdir(os.path.join(DATASET_PATH, 'train'))
st.write("📂 Categories found:", categories)

# Load or train models
existing_model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
if not existing_model_files:
    st.warning("⚠️ No trained models found. Training models now...")
    x_train, y_train = load_data_from_directory(os.path.join(DATASET_PATH, 'train'), categories, IMAGE_SIZE, MAX_IMAGES_PER_CLASS)
    x_valid, y_valid = load_data_from_directory(os.path.join(DATASET_PATH, 'valid'), categories, IMAGE_SIZE, MAX_IMAGES_PER_CLASS)
    models, accuracies = train_and_save_models(x_train, y_train, x_valid, y_valid, categories)
else:
    st.success("✅ Pre-trained models found. Loading them...")
    models = load_saved_models()
    accuracies = load_accuracies_from_json()
    if not accuracies:
        st.warning("⚠️ Accuracy file not found. Please retrain the models to regenerate.")

# Accuracy chart
st.subheader("📊 Validation Accuracy")
acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"]).sort_values(by="Accuracy", ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(acc_df["Model"], acc_df["Accuracy"] * 100, color="skyblue")
best_model = acc_df.iloc[0]["Model"]
for i, bar in enumerate(bars):
    if acc_df["Model"].iloc[i] == best_model:
        bar.set_color("green")
ax.set_title("Model Performance")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
st.pyplot(fig)

# Model selection
st.success(f"🏆 Best Model: {best_model} ({accuracies[best_model] * 100:.2f}%)")
selected_models = st.multiselect("🔘 Choose model(s) for prediction:", options=list(models.keys()), default=[best_model])

# Upload and predict
st.subheader("📸 Upload Fish Image for Prediction")
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    img = imread(uploaded_file)
    img_resized = resize(img, IMAGE_SIZE).flatten().reshape(1, -1)

    for model_name in selected_models:
        model = models[model_name]
        pred_index = model.predict(img_resized)[0]
        pred_label = categories[pred_index]
        st.info(f"🔍 Prediction using {model_name}: **{pred_label}**")

st.markdown("---")
