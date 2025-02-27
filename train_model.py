import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Loading Data
df = pd.read_csv("augmented_battery_data.csv")

# Identification of required columns
expected_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)", "Battery Type"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"Data set is missing required columns: {df.columns}")

X = df.drop(columns=["Battery Type"])
y = df["Battery Type"]

# label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Training and test data partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model training (parameter tuning)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and encoder
joblib.dump(model, "battery_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model training completed and saved.")
