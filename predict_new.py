import numpy as np
import joblib
import pandas as pd

# モデル、スケーラー、ラベルエンコーダーをロード
model = joblib.load("battery_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 新しいデータ（例：バッテリーの特性を入力）
new_data = np.array([[15.0, 35, 1.2, 3.5, 9.5]])

# DataFrame に変換（学習時の特徴量と同じカラム名を持つ）
X_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
new_data_df = pd.DataFrame(new_data, columns=X_columns)

# 標準化
new_data_scaled = scaler.transform(new_data_df)

# 予測
predicted_label = model.predict(new_data_scaled)
predicted_battery_type = label_encoder.inverse_transform(predicted_label)

# 結果を表示
print("予測されたバッテリータイプ:", predicted_battery_type[0])
