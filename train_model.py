import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# データの読み込み
df = pd.read_csv("augmented_battery_data.csv")

# 特徴量とターゲット変数の分離
X = df.drop(columns=["Battery Type"])
y = df["Battery Type"]

# ラベルエンコーディング
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 訓練・テストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# スケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# モデルの学習
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# モデルとエンコーダーを保存
joblib.dump(model, "battery_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("Model training completed and saved.")
