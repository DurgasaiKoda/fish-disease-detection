import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib
import os

# Load Data with Limit
@st.cache_data(show_spinner=True)
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
            except Exception as e:
                st.warning(f"Failed to load: {img_path}")
    return np.array(data), np.array(labels)

# Define Models
def get_models():
    return {
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True)),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

# UI Setup
st.set_page_config(page_title="Fish Disease Classifier", layout="wide")
st.title("🐟 Fish Disease Classification App")

# Dataset Path
dataset_path = 'dataset'
st.write("🔍 Loading dataset from:", dataset_path)

# Load categories
try:
    categories = os.listdir(os.path.join(dataset_path, 'train'))
    st.write("✅ Categories found:", categories)
except Exception as e:
    st.error("🚫 Error reading categories. Please check your dataset path.")
    st.stop()

# Load dataset (limit to 100 images per category for speed)
with st.spinner("Loading image data..."):
    x_train, y_train = load_data_from_directory(os.path.join(dataset_path, 'train'), categories)
    x_valid, y_valid = load_data_from_directory(os.path.join(dataset_path, 'valid'), categories)
    x_test, y_test   = load_data_from_directory(os.path.join(dataset_path, 'test'), categories)

st.success("✅ Image data loaded!")
st.write("Train samples:", x_train.shape[0])
st.write("Validation samples:", x_valid.shape[0])
st.write("Test samples:", x_test.shape[0])

# Train models
models = get_models()
accuracies = {}

with st.spinner("🧠 Training models..."):
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_val_pred = model.predict(x_valid)
        acc = accuracy_score(y_valid, y_val_pred)
        accuracies[name] = acc

# Best model selection
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
joblib.dump(best_model, 'fish_model.pkl')

# Display accuracy chart
st.subheader("📊 Model Performance on Validation Set")
accuracy_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
accuracy_df = accuracy_df.sort_values(by="Accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(accuracy_df["Model"], accuracy_df["Accuracy"] * 100, color="skyblue")
for i, bar in enumerate(bars):
    if accuracy_df["Model"].iloc[i] == best_model_name:
        bar.set_color("green")
ax.set_title("Validation Accuracy of Models")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
st.pyplot(fig)

# Show best model
st.success(f"🏆 Best Model: {best_model_name} ({accuracies[best_model_name]*100:.2f}%)")

# Select model(s)
st.subheader("🧠 Choose Model(s) to Use for Prediction")
selected_models = st.multiselect(
    "Select model(s)", options=list(models.keys()), default=[best_model_name]
)

# Upload & predict
st.subheader("📸 Upload Fish Image for Prediction")
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = imread(uploaded_file)
    img_resized = resize(img, (64, 64)).flatten().reshape(1, -1)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    for model_name in selected_models:
        model = models[model_name]
        pred_index = model.predict(img_resized)[0]
        pred_label = categories[pred_index]
        st.info(f"🔍 **Prediction using {model_name}**: {pred_label}")

st.markdown("---")
