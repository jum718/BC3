import streamlit as st
import numpy as np
import joblib
import pandas as pd

# モデル・スケーラー・ラベルエンコーダーのロード（セッションごとに実行）
@st.cache_resource
def load_model():
    model = joblib.load("battery_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

# ユーザー入力
st.sidebar.header("Enter Battery Specifications")
battery_weight = st.sidebar.number_input("Battery Weight (kg)", min_value=5.0, max_value=50.0, value=15.0)
battery_density = st.sidebar.number_input("Battery Density (Wh/kg)", min_value=10, max_value=100, value=35)
battery_acid = st.sidebar.number_input("Battery Acid (pH)", min_value=0.5, max_value=2.0, value=1.2)
plastic_weight = st.sidebar.number_input("Plastic Weight (kg)", min_value=0.5, max_value=10.0, value=3.5)
lead_weight = st.sidebar.number_input("Lead Weight (kg)", min_value=1.0, max_value=20.0, value=9.5)

# セッション変数をリセットする関数
def reset_prediction():
    st.session_state.prediction = None

# 予測ボタン
if st.sidebar.button("Predict Battery Type", on_click=reset_prediction):
    # 入力データを配列に変換
    new_data = np.array([[battery_weight, battery_density, battery_acid, plastic_weight, lead_weight]])

    # DataFrame に変換（特徴量のカラム名を保持）
    X_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
    new_data_df = pd.DataFrame(new_data, columns=X_columns)

    # 標準化
    new_data_scaled = scaler.transform(new_data_df)

    # 予測
    predicted_label = model.predict(new_data_scaled)
    predicted_battery_type = label_encoder.inverse_transform(predicted_label)

    # 予測結果をセッション変数に保存
    st.session_state.prediction = predicted_battery_type[0]

# 結果表示（予測が実行された後のみ表示）
if "prediction" in st.session_state and st.session_state.prediction:
    st.subheader("Prediction Result")
    st.write(f"Predicted Battery Type: **{st.session_state.prediction}**")
