import streamlit as st
import numpy as np
import joblib
import pandas as pd
import random

# Title
st.title("Material Type Classifier Demo")

# Model Loading
try:
    model = joblib.load("Type.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please check the model directory.")
    st.stop()

# Session state initialization
if "predictions" not in st.session_state:
    st.session_state.predictions = {"Plastic (PP/ABS)": 0, "Lead (Pb)": 0, "Sulfuric Acid (H2SO4)": 0}
if "history" not in st.session_state:
    st.session_state.history = []
if "predicted_material_type" not in st.session_state:
    st.session_state.predicted_material_type = ""

# Sidebar header
st.sidebar.header("Material Specifications received from Laser")

# Random Value Generation Functions
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
        "acoustic": random.choice(["High", "Medium", "Low"]),
        "composition": random.choice(["C, H, O", "H, S, O", "Pb"]),
        "physical_state": random.choice(["Liquid", "Solid"])
    }

if "random_values" not in st.session_state:
    st.session_state.random_values = generate_random_values()

if st.sidebar.button("Generate Random Values"):
    st.session_state.random_values = generate_random_values()

# User input fields
density = st.sidebar.number_input("Density (g/cm3)", 1.0, 10.0, st.session_state.random_values["density"])
xray = st.sidebar.number_input("X-ray Absorption (HU)", 50, 500, st.session_state.random_values["xray"])
energy = st.sidebar.number_input("Energy Spectrum (keV)", 1.0, 10.0, st.session_state.random_values["energy"])
magnetic = st.sidebar.selectbox("Magnetic Response", ["Non-magnetic", "Magnetic"], index=["Non-magnetic", "Magnetic"].index(st.session_state.random_values["magnetic"]))
weight = st.sidebar.number_input("Weight Contribution", 0.1, 5.0, st.session_state.random_values["weight"])
thermal = st.sidebar.number_input("Thermal Conductivity", 10.0, 200.0, st.session_state.random_values["thermal"])
infrared = st.sidebar.selectbox("Infrared (IR) Signature", ["Infrared", "Visible", "None"], index=["Infrared", "Visible", "None"].index(st.session_state.random_values["infrared"]))
uv = st.sidebar.selectbox("UV Reactivity", ["Reactive", "Non-reactive"], index=["Reactive", "Non-reactive"].index(st.session_state.random_values["uv"]))
electrical = st.sidebar.number_input("Electrical Conductivity", 0.1, 10.0, st.session_state.random_values["electrical"])
acoustic = st.sidebar.selectbox("Acoustic Response", ["High", "Medium", "Low"], index=["High", "Medium", "Low"].index(st.session_state.random_values["acoustic"]))
composition = st.sidebar.selectbox("Elemental Composition", ["C, H, O", "H, S, O", "Pb"], index=["C, H, O", "H, S, O", "Pb"].index(st.session_state.random_values["composition"]))
physical_state = st.sidebar.selectbox("Physical State", ["Liquid", "Solid"], index=["Liquid", "Solid"].index(st.session_state.random_values["physical_state"]))

# Prediction result display
st.subheader("Prediction Result")

# Display "-" if there is no prediction result
display_text = st.session_state.predicted_material_type if st.session_state.predicted_material_type else "-"

st.markdown(
    f'<p style="font-size:24px; color:#F06060;"><b>Predicted Material Type: {display_text}</b></p>',
    unsafe_allow_html=True
)

# Predict button functionality
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
        "Acoustic Response": [2 if acoustic == "High" else 1 if acoustic == "Medium" else 0],
        "Elemental Composition_" + composition: [1],
        "Physical State_" + physical_state: [1]
    })
    
    new_data = new_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    try:
        new_data_scaled = scaler.transform(new_data)
        predicted_label = model.predict(new_data_scaled)
        st.session_state.predicted_material_type = label_encoder.inverse_transform(predicted_label)[0]

        # Update predictions count
        st.session_state.predictions[st.session_state.predicted_material_type] = (
            st.session_state.predictions.get(st.session_state.predicted_material_type, 0) + 1
        )

        # Result Update Update**
        st.rerun()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Count display of forecast results
st.subheader("Prediction Counts")
df_counts = pd.DataFrame(list(st.session_state.predictions.items()), columns=["Predicted Type", "Count"])
st.dataframe(df_counts)

# Reset button
if st.button("Reset Counts & History"):
    st.session_state.predictions = {"Plastic (PP/ABS)": 0, "Lead (Pb)": 0, "Sulfuric Acid (H2SO4)": 0}
    st.session_state.history = []
    st.session_state.predicted_material_type = ""
    st.success("Counts and history have been reset.")
    st.rerun()
