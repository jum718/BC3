import numpy as np
import joblib
import pandas as pd
import argparse

# Setting Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--density", type=float, required=True, help="Material Density (g/cm3)")
parser.add_argument("--xray", type=int, required=True, help="X-ray Absorption (HU)")
parser.add_argument("--energy", type=float, required=True, help="Energy Spectrum (keV)")
parser.add_argument("--magnetic", type=str, required=True, choices=["Non-magnetic", "Magnetic"], help="Magnetic Response")
parser.add_argument("--weight", type=float, required=True, help="Weight Contribution")
parser.add_argument("--thermal", type=float, required=True, help="Thermal Conductivity")
parser.add_argument("--infrared", type=str, required=True, choices=["Infrared", "Visible", "None"], help="Infrared (IR) Signature")
parser.add_argument("--uv", type=str, required=True, choices=["Reactive", "Non-reactive"], help="UV Reactivity")
parser.add_argument("--electrical", type=float, required=True, help="Electrical Conductivity")
parser.add_argument("--acoustic", type=str, required=True, choices=["High", "Medium", "Low"], help="Acoustic Response")
args = parser.parse_args()

# Load models, scalers, and encoders
model = joblib.load("Type.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Creation of input data
new_data = pd.DataFrame({
    "Density (g/cm3)": [args.density],
    "X-ray Absorption (HU)": [args.xray],
    "Energy Spectrum (keV)": [args.energy],
    "Magnetic Response": [1 if args.magnetic == "Magnetic" else 0],
    "Weight Contribution": [args.weight],
    "Thermal Conductivity": [args.thermal],
    "Infrared (IR) Signature": [0 if args.infrared == "Infrared" else 1 if args.infrared == "Visible" else 2],
    "UV Reactivity": [1 if args.uv == "Reactive" else 0],
    "Electrical Conductivity": [args.electrical],
    "Acoustic Response": [2 if args.acoustic == "High" else 1 if args.acoustic == "Medium" else 0]
})

# Scaling Application
new_data_scaled = scaler.transform(new_data)

# Implementation of forecasts
predicted_label = model.predict(new_data_scaled)
predicted_material_type = label_encoder.inverse_transform(predicted_label)

# Display Results
print("Predicted Material Type:", predicted_material_type[0])
