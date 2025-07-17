# 🐟 Fish Disease Detection Web App using Streamlit

This project is a **Streamlit-based web application** for detecting diseases in fish using machine learning image classification models. It allows users to upload an image of a fish and get predictions on whether it is **healthy or diseased**, along with identifying the **specific disease type**.

The application compares multiple machine learning algorithms (SVM, Random Forest, Logistic Regression, Decision Tree, XGBoost) and selects the best-performing model for prediction.

---

## 🔗 Features

- Upload a fish image to detect if it is **healthy** or **diseased**
- Identify the **type of disease** based on the image
- Trains and compares **five ML algorithms**
- Automatically caches models and accuracies for future sessions
- Displays **model-wise accuracy graph**
- Allows user to **manually select model** for prediction
- Uses `joblib` to save and load trained models
- Handles large datasets efficiently by flattening resized images

---

## 📁 Project Structure
```bash
fish_disease_detector/
├── app.py # Streamlit app entry point
├── train_models.py # (Optional) Script to train and save models
├── requirements.txt # Python dependencies
├── dataset/ # Image dataset (train/valid/test folders)
│ ├── train/
│ │ ├── Healthy/
│ │ └── Disease1/...
│ ├── valid/
│ │ ├── Healthy/
│ │ └── Disease1/...
│ └── test/
│ ├── Healthy/
│ └── Disease1/...
├── models/ # Saved trained models (as .pkl)
│ ├── SVM.pkl
│ ├── Random Forest.pkl
│ ├── Logistic Regression.pkl
│ ├── Decision Tree.pkl
│ └── XGBoost.pkl
├── accuracies.json # Saved validation accuracies
├── utils.py # (Optional) Reusable functions
└── README.md # You're reading it!
```

---

## ⚙️ Technologies Used

- 🐍 Python
- 📦 Streamlit
- 📊 Scikit-learn
- 🌲 XGBoost
- 🖼 scikit-image
- 🧠 NumPy, Pandas
- 📦 joblib for model persistence
- 📈 Matplotlib for graph plotting

---

## ✅ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/fish_disease_detector.git
cd fish_disease_detector
```
2. **Create virtual environment (optional but recommended):**

```bash
python -m venv venv
venv\Scripts\activate
#On IOS: source venv/bin/activate  
```
3. **Install dependencies:**

```bash
pip install -r requirements.txt
```
----

## 🚀 Run the App

```bash
streamlit run app.py
```
This will open the app in your default web browser.

## 📂 Dataset Structure

The dataset folder should be structured as follows:

```bash
dataset/
├── train/
│   ├── Healthy/
│   └── Disease1/
├── valid/
│   ├── Healthy/
│   └── Disease1/
└── test/
    ├── Healthy/
    └── Disease1/
```
Each subfolder (like `Healthy`, `Gill Disease`, `Tail Rot`, etc.) represents a class label.

## 💡 First-Time Run Notes

- If no trained models exist in the `models/` folder, the app will train and save them.
- After the initial training, models are loaded from the `.pkl` files to avoid retraining.
- Accuracies are saved to `accuracies.json` and reused in future sessions.

## 🛠 How It Works

- All images are resized to `(64, 64)` and flattened.
- Models are trained using the `train` dataset.
- Validation set is used to calculate accuracy and select the best model.
- The best model is used for prediction on user-uploaded images.
- You can manually choose a model via a dropdown.
- The accuracy graph helps visualize performance comparison.

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## ✨ Acknowledgements

- Streamlit for making web app development simple.
- scikit-learn and XGBoost for model support.
- Open datasets for fish disease classification.






