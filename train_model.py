import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Loading Data
df = pd.read_csv("BC3-2/material_properties.csv")

# Identification of required columns
expected_columns = ["Density (g/cm3)", "X-ray Absorption (HU)", "Energy Spectrum (keV)", "Elemental Composition", "Magnetic Response", "Weight Contribution", "Physical State", "Thermal Conductivity", "Infrared (IR) Signature", "UV Reactivity", "Electrical Conductivity", "Acoustic Response"]
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"Data set is missing required columns: {df.columns}")

X = df.drop(columns=["Type"])
y = df["Type"]

# Magnetic Response列を数値に変換
X["Magnetic Response"] = X["Magnetic Response"].map({"Non-magnetic": 0, "Magnetic": 1})

# Elemental Composition列をOne-Hotエンコーディング
X = pd.get_dummies(X, columns=["Elemental Composition"], dummy_na=False)

# Physical State列をOne-Hotエンコーディング
X = pd.get_dummies(X, columns=["Physical State"], dummy_na=False)

# Infrared (IR) Signature列を数値に変換
X["Infrared (IR) Signature"] = X["Infrared (IR) Signature"].map({"Infrared": 0, "Visible": 1, "None": 2})

# UV Reactivity列を数値に変換
X["UV Reactivity"] = X["UV Reactivity"].map({"Reactive": 1, "Non-reactive": 0})

# Acoustic Response列を数値に変換
X["Acoustic Response"] = X["Acoustic Response"].map({"High": 2, "Medium": 1, "Low": 0})

# 欠損値の処理（例：平均値で補完）
X_numeric = X.select_dtypes(include=['number'])
X[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())

# 数値として解釈できない文字列の処理（例：行を削除）
X = X[pd.to_numeric(X["Density (g/cm3)"], errors='coerce').notna()]

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
joblib.dump(model, "BC3-2/Type.pkl")
joblib.dump(scaler, "BC3-2/scaler.pkl")
joblib.dump(label_encoder, "BC3-2/label_encoder.pkl")

print("Model training completed and saved.")
