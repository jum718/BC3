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

# タイトル
st.title("Battery Type Classifier")

# ユーザー入力方法の選択
option = st.radio("Choose input method:", ("Manual Input", "Upload CSV File"))

# --- **1. 手入力モード** ---
if option == "Manual Input":
    st.sidebar.header("Enter Battery Specifications")
    
    # 手動入力用のスライダー
    battery_weight = st.sidebar.number_input("Battery Weight (kg)", min_value=5.0, max_value=50.0, value=15.0)
    battery_density = st.sidebar.number_input("Battery Density (Wh/kg)", min_value=10, max_value=100, value=35)
    battery_acid = st.sidebar.number_input("Battery Acid (pH)", min_value=0.5, max_value=2.0, value=1.2)
    plastic_weight = st.sidebar.number_input("Plastic Weight (kg)", min_value=0.5, max_value=10.0, value=3.5)
    lead_weight = st.sidebar.number_input("Lead Weight (kg)", min_value=1.0, max_value=20.0, value=9.5)

    # 予測ボタン
    if st.sidebar.button("Predict Battery Type"):
        # 入力データを配列に変換
        new_data = np.array([[battery_weight, battery_density, battery_acid, plastic_weight, lead_weight]])

        # DataFrame に変換
        X_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
        new_data_df = pd.DataFrame(new_data, columns=X_columns)

        # 標準化
        new_data_scaled = scaler.transform(new_data_df)

        # 予測
        predicted_label = model.predict(new_data_scaled)
        predicted_battery_type = label_encoder.inverse_transform(predicted_label)

        # 結果表示
        st.subheader("Prediction Result")
        st.write(f"Predicted Battery Type: **{predicted_battery_type[0]}**")

# --- **2. CSVアップロードモード** ---
elif option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # CSVファイルを読み込む
        df_uploaded = pd.read_csv(uploaded_file)

        # 必要なカラムが揃っているか確認
        required_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
        if all(col in df_uploaded.columns for col in required_columns):
            # 標準化
            df_scaled = scaler.transform(df_uploaded)

            # 予測
            predictions = model.predict(df_scaled)
            df_uploaded["Predicted Battery Type"] = label_encoder.inverse_transform(predictions)

            # 結果を表示
            st.subheader("Prediction Results")
            st.write(df_uploaded)

            # 結果をCSVとしてダウンロードできるようにする
            csv = df_uploaded.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="battery_predictions.csv", mime="text/csv")

        else:
            st.error("Uploaded CSV must contain the following columns: " + ", ".join(required_columns))
