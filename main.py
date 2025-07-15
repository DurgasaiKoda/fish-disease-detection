# main.py
import os
from data_loader import load_data_from_directory
from models import get_models
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

dataset_path = 'dataset'
categories = os.listdir(os.path.join(dataset_path, 'train'))

print("Loading data...")
x_train, y_train = load_data_from_directory(os.path.join(dataset_path, 'train'), categories)
x_valid, y_valid = load_data_from_directory(os.path.join(dataset_path, 'valid'), categories)
x_test, y_test   = load_data_from_directory(os.path.join(dataset_path, 'test'), categories)

models = get_models()
accuracies = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_valid)
    acc = accuracy_score(y_valid, y_val_pred)
    accuracies[name] = acc
    print(f"{name} Validation Accuracy: {acc*100:.2f}%")
    print(classification_report(y_valid, y_val_pred, target_names=categories))

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

y_test_pred = best_model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n🏁 Final Test Accuracy of {best_model_name}: {test_acc*100:.2f}%")
print(classification_report(y_test, y_test_pred, target_names=categories))

# Plotting
plt.bar(accuracies.keys(), [v*100 for v in accuracies.values()], color='skyblue')
plt.ylabel("Validation Accuracy (%)")
plt.title("Model Comparison")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
