import numpy as np
import joblib
import pandas as pd
import argparse

# Setting Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--weight", type=float, required=True, help="Battery Weight (kg)")
parser.add_argument("--density", type=int, required=True, help="Battery Density (Wh/kg)")
parser.add_argument("--acid", type=float, required=True, help="Battery Acid (pH)")
parser.add_argument("--plastic", type=float, required=True, help="Plastic Weight (kg)")
parser.add_argument("--lead", type=float, required=True, help="Lead Weight (kg)")
args = parser.parse_args()

# Load models, scalers, and encoders
model = joblib.load("battery_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# input data
new_data = np.array([[args.weight, args.density, args.acid, args.plastic, args.lead]])

# Convert to DataFrame
X_columns = ["Battery Weight (kg)", "Battery Density (Wh/kg)", "Battery Acid (pH)", "Plastic Weight (kg)", "Lead Weight (kg)"]
new_data_df = pd.DataFrame(new_data, columns=X_columns)

# standardization
new_data_scaled = scaler.transform(new_data_df)

# prediction
predicted_label = model.predict(new_data_scaled)
predicted_battery_type = label_encoder.inverse_transform(predicted_label)

# Show Results
print("Predicted battery type:", predicted_battery_type[0])
