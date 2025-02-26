import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# データの読み込み
df = pd.read_csv("augmented_battery_data.csv")

# 特徴量とターゲット変数の分離
X = df.drop(columns=["Battery Type"])
y = df["Battery Type"]

# エンコードをロード
label_encoder = joblib.load("label_encoder.pkl")
y_encoded = label_encoder.transform(y)

# 訓練データとテストデータに分割
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# スケーラーをロードして標準化
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# モデルをロード
model = joblib.load("battery_model.pkl")

# 予測
y_pred = model.predict(X_test_scaled)

# 評価結果を表示
accuracy = accuracy_score(y_test, y_pred)
print(f"モデルの精度: {accuracy:.2f}")
print("\n分類レポート:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\n混同行列:\n", confusion_matrix(y_test, y_pred))
