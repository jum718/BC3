import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading Data
df = pd.read_csv("augmented_battery_data.csv")

X = df.drop(columns=["Battery Type"])
y = df["Battery Type"]

# Load Encoding
label_encoder = joblib.load("label_encoder.pkl")
y_encoded = label_encoder.transform(y)

# Split into training and test data
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Load Scaler
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# Load Model
model = joblib.load("battery_model.pkl")

# Prediction
y_pred = model.predict(X_test_scaled)

# Display evaluation results
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualization of confusion matrices
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
