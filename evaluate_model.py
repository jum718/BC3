import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# データの読み込み
df = pd.read_csv("augmented_battery_data.csv")

X = df.drop(columns=["Battery Type"])
y = df["Battery Type"]

# エンコードのロード
label_encoder = joblib.load("label_encoder.pkl")
y_encoded = label_encoder.transform(y)

# 訓練データとテストデータに分割
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# スケーラーをロード
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

# 混同行列の可視化
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
