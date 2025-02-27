import streamlit as st
import numpy as np
import joblib
import pandas as pd
import random
import io

# タイトル
st.title("Material Type Classifier")

# モデルのロード
try:
    model = joblib.load("Type.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please check the model directory.")
    st.stop()

# セッションステートの初期化
if "predictions" not in st.session_state:
    st.session_state.predictions = {}
if "history" not in st.session_state:
    st.session_state.history = []

# ランダム値生成関数
def generate_random_values():
    return {
        "density": round(random.uniform(1.0, 10.0), 2),
        "xray": random.randint(50, 500),
        "energy": round(random.uniform(1.0, 10.0), 2),
        "magnetic": random.choice(["Non-magnetic", "Magnetic"]),
        "weight": round(random.uniform(0.1, 5.0), 2),
        "thermal": round(random.uniform(10.0, 200.0), 2),
        "infrared": random.choice(["Infrared", "Visible", "None"]),
        "uv": random.choice(["Reactive", "Non-reactive"]),
        "electrical": round(random.uniform(0.1, 10.0), 2),
        "acoustic": random.choice(["High", "Medium", "Low"])
    }

if "random_values" not in st.session_state:
    st.session_state.random_values = generate_random_values()

if st.sidebar.button("Generate Random Values"):
    st.session_state.random_values = generate_random_values()

# ユーザー入力
st.sidebar.header("Enter Material Specifications")
density = st.sidebar.number_input("Density (g/cm3)", min_value=1.0, max_value=10.0, value=st.session_state.random_values["density"])
xray = st.sidebar.number_input("X-ray Absorption (HU)", min_value=50, max_value=500, value=st.session_state.random_values["xray"])
energy = st.sidebar.number_input("Energy Spectrum (keV)", min_value=1.0, max_value=10.0, value=st.session_state.random_values["energy"])
magnetic = st.sidebar.selectbox("Magnetic Response", ["Non-magnetic", "Magnetic"], index=["Non-magnetic", "Magnetic"].index(st.session_state.random_values["magnetic"]))
weight = st.sidebar.number_input("Weight Contribution", min_value=0.1, max_value=5.0, value=st.session_state.random_values["weight"])
thermal = st.sidebar.number_input("Thermal Conductivity", min_value=10.0, max_value=200.0, value=st.session_state.random_values["thermal"])
infrared = st.sidebar.selectbox("Infrared (IR) Signature", ["Infrared", "Visible", "None"], index=["Infrared", "Visible", "None"].index(st.session_state.random_values["infrared"]))
uv = st.sidebar.selectbox("UV Reactivity", ["Reactive", "Non-reactive"], index=["Reactive", "Non-reactive"].index(st.session_state.random_values["uv"]))
electrical = st.sidebar.number_input("Electrical Conductivity", min_value=0.1, max_value=10.0, value=st.session_state.random_values["electrical"])
acoustic = st.sidebar.selectbox("Acoustic Response", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(st.session_state.random_values["acoustic"]))

# 予測ボタン
if st.sidebar.button("Predict Material Type"):
    new_data = pd.DataFrame({
        "Density (g/cm3)": [density],
        "X-ray Absorption (HU)": [xray],
        "Energy Spectrum (keV)": [energy],
        "Magnetic Response": [1 if magnetic == "Magnetic" else 0],
        "Weight Contribution": [weight],
        "Thermal Conductivity": [thermal],
        "Infrared (IR) Signature": [0 if infrared == "Infrared" else 1 if infrared == "Visible" else 2],
        "UV Reactivity": [1 if uv == "Reactive" else 0],
        "Electrical Conductivity": [electrical],
        "Acoustic Response": [2 if acoustic == "High" else 1 if acoustic == "Medium" else 0]
    })

    try:
        new_data_scaled = scaler.transform(new_data)
        predicted_label = model.predict(new_data_scaled)
        predicted_material_type = label_encoder.inverse_transform(predicted_label)[0]

        if predicted_material_type in st.session_state.predictions:
            st.session_state.predictions[predicted_material_type] += 1
        else:
            st.session_state.predictions[predicted_material_type] = 1

        new_entry = {
            "Density (g/cm3)": density,
            "X-ray Absorption (HU)": xray,
            "Energy Spectrum (keV)": energy,
            "Magnetic Response": magnetic,
            "Weight Contribution": weight,
            "Thermal Conductivity": thermal,
            "Infrared (IR) Signature": infrared,
            "UV Reactivity": uv,
            "Electrical Conductivity": electrical,
            "Acoustic Response": acoustic,
            "Predicted Material Type": predicted_material_type
        }
        st.session_state.history.append(new_entry)

        st.subheader("Prediction Result")
        st.write(f"Predicted Material Type: **{predicted_material_type}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# 予測回数の表示
st.subheader("Prediction Counts")
if st.session_state.predictions:
    df_counts = pd.DataFrame(list(st.session_state.predictions.items()), columns=["Material Type", "Count"])
    st.dataframe(df_counts)
else:
    st.write("No predictions yet.")

# CSV ダウンロード
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    csv_buffer = io.StringIO()
    df_history.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="material_predictions.csv",
        mime="text/csv"
    )

# リセットボタン
if st.button("Reset Counts & History"):
    st.session_state.predictions = {}
    st.session_state.history = []
    st.success("Counts and history have been reset.")
    st.rerun()
