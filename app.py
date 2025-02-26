import streamlit as st
import numpy as np
import joblib
import pandas as pd

# タイトル
st.title("Battery Type Classifier")

# モデルのロード
try:
    model = joblib.load("battery_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("モデルファイルが見つかりません。train_model.py を実行してください。")
    st.stop()

# セッションステートの初期化
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

# ユーザー入力
st.sidebar.header("Enter Battery Specifications")
battery_weight = st.sidebar.number_input("Battery Weight (kg)", min_value=5.0, max_value=50.0, value=15.0)
battery_density = st.sidebar.number_input("Battery Density (Wh/kg)", min_value=10, max_value=100, value=35)
battery_acid = st.sidebar.number_input("Battery Acid (pH)", min_value=0.5, max_value=2.0, value=1.2)
plastic_weight = st.sidebar.number_input("Plastic Weight (kg)", min_value=0.5, max_value=10.0, value=3.5)
lead_weight = st.sidebar.number_input("Lead Weight (kg)", min_value=1.0, max_value=20.0, value=9.5)

# 予測ボタン
if st.sidebar.button("Predict Battery Type"):
    new_data = np.array([[battery_weight, battery_density, battery_acid, plastic_weight, lead_weight]])
    X_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
    new_data_df = pd.DataFrame(new_data, columns=X_columns)

    try:
        new_data_scaled = scaler.transform(new_data_df)
        predicted_label = model.predict(new_data_scaled)
        predicted_battery_type = label_encoder.inverse_transform(predicted_label)[0]

        # カウントを更新
        if predicted_battery_type in st.session_state.predictions:
            st.session_state.predictions[predicted_battery_type] += 1
        else:
            st.session_state.predictions[predicted_battery_type] = 1

        st.subheader("Prediction Result")
        st.write(f"Predicted Battery Type: **{predicted_battery_type}**")

    except Exception as e:
        st.error(f"予測中にエラーが発生しました: {e}")

# 結果の統計表示
st.subheader("Prediction Counts")
if st.session_state.predictions:
    df_counts = pd.DataFrame(list(st.session_state.predictions.items()), columns=["Battery Type", "Count"])
    st.dataframe(df_counts)
else:
    st.write("まだ予測結果はありません。")

# リセットボタン
if st.button("Reset Counts"):
    st.session_state.predictions = {}
    st.success("カウントをリセットしました。")
