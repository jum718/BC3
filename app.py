import streamlit as st
import numpy as np
import joblib
import pandas as pd
import random
import io

# Title
st.title("Battery Type Classifier")

# Load the model
try:
    model = joblib.load("battery_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py.")
    st.stop()

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = {}

if "history" not in st.session_state:
    st.session_state.history = []

# Function to generate random values
def generate_random_values():
    return {
        "battery_weight": round(random.uniform(5.0, 50.0), 1),
        "battery_density": random.randint(10, 100),
        "battery_acid": round(random.uniform(0.5, 2.0), 2),
        "plastic_weight": round(random.uniform(0.5, 10.0), 1),
        "lead_weight": round(random.uniform(1.0, 20.0), 1)
    }

# Update session state when "Generate Random Values" is clicked
if "random_values" not in st.session_state:
    st.session_state.random_values = generate_random_values()

if st.sidebar.button("Generate Random Values"):
    st.session_state.random_values = generate_random_values()

# User input
st.sidebar.header("Enter Battery Specifications")
battery_weight = st.sidebar.number_input(
    "Battery Weight (kg)", min_value=5.0, max_value=50.0, value=st.session_state.random_values["battery_weight"]
)
battery_density = st.sidebar.number_input(
    "Battery Density (Wh/kg)", min_value=10, max_value=100, value=st.session_state.random_values["battery_density"]
)
battery_acid = st.sidebar.number_input(
    "Battery Acid (pH)", min_value=0.5, max_value=2.0, value=st.session_state.random_values["battery_acid"]
)
plastic_weight = st.sidebar.number_input(
    "Plastic Weight (kg)", min_value=0.5, max_value=10.0, value=st.session_state.random_values["plastic_weight"]
)
lead_weight = st.sidebar.number_input(
    "Lead Weight (kg)", min_value=1.0, max_value=20.0, value=st.session_state.random_values["lead_weight"]
)

# Predict button
if st.sidebar.button("Predict Battery Type"):
    new_data = np.array([[battery_weight, battery_density, battery_acid, plastic_weight, lead_weight]])
    X_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
    new_data_df = pd.DataFrame(new_data, columns=X_columns)

    try:
        new_data_scaled = scaler.transform(new_data_df)
        predicted_label = model.predict(new_data_scaled)
        predicted_battery_type = label_encoder.inverse_transform(predicted_label)[0]

        # Update count
        if predicted_battery_type in st.session_state.predictions:
            st.session_state.predictions[predicted_battery_type] += 1
        else:
            st.session_state.predictions[predicted_battery_type] = 1

        # Store input and result in history
        new_entry = {
            "Battery Weight (kg)": battery_weight,
            "Battery Density (Wh/kg)": battery_density,
            "Battery Acid (pH)": battery_acid,
            "Plastic Weight (kg)": plastic_weight,
            "Lead Weight (kg)": lead_weight,
            "Predicted Battery Type": predicted_battery_type
        }
        st.session_state.history.append(new_entry)

        st.subheader("Prediction Result")
        st.write(f"Predicted Battery Type: **{predicted_battery_type}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Display prediction counts
st.subheader("Prediction Counts")
if st.session_state.predictions:
    df_counts = pd.DataFrame(list(st.session_state.predictions.items()), columns=["Battery Type", "Count"])
    st.dataframe(df_counts)
else:
    st.write("No predictions yet.")

# CSV Download
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)

    # Convert DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    df_history.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="battery_predictions.csv",
        mime="text/csv"
    )

# Reset button with immediate effect
if st.button("Reset Counts & History"):
    st.session_state.predictions = {}
    st.session_state.history = []
    st.success("Counts and history have been reset.")

    # Force immediate UI update
    st.experimental_rerun()
