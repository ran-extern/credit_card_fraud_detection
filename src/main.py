import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('weights/models/random_forest_model.pkl')
scaler = joblib.load('weights/scaler/scaler.pkl')

# Define feature names (excluding the label 'fraud')
FEATURE_NAMES = [
    'distance_from_home',
    'distance_from_last_transaction',
    'ratio_to_median_purchase_price',
    'repeat_retailer',
    'used_chip',
    'used_pin_number',
    'online_order'
]


# Prediction function
def predict_transaction(distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order):
    # Input validation
    try:
        distance_from_home = float(distance_from_home)
        distance_from_last_transaction = float(distance_from_last_transaction)
        ratio_to_median_purchase_price = float(ratio_to_median_purchase_price)

        if distance_from_home <= 0:
            return {
                "status": "error",
                "message": "❌ Distance from home must be positive!"
            }
        if distance_from_last_transaction <= 0:
            return {
                "status": "error",
                "message": "❌ Distance from last transaction must be positive!"
            }
        if ratio_to_median_purchase_price <= 0:
            return {
                "status": "error",
                "message": "❌ Ratio to median price must be positive!"
            }
    except ValueError:
        return {
            "status": "error",
            "message": "❌ Enter valid numbers for distances and ratio!"
        }

    # Map Yes/No to 1/0
    repeat_retailer = 1 if repeat_retailer == 'Yes' else 0
    used_chip = 1 if used_chip == 'Yes' else 0
    used_pin_number = 1 if used_pin_number == 'Yes' else 0
    online_order = 1 if online_order == 'Yes' else 0

    # Prepare features as a DataFrame
    features = pd.DataFrame(
        [[distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order]],
        columns=FEATURE_NAMES
    )

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)[0]

    # Format output
    if prediction == 0:
        return {
            "status": "success",
            "message": "✅ Normal Transaction",
            "details": "Appears legitimate."
        }
    else:
        return {
            "status": "warning",
            "message": "⚠️ Fraudulent Transaction",
            "details": "Potential fraud detected."
        }


# Format output as plain text
def format_output(result):
    return f"{result['message']}\n{result['details']}"


# Create Interface
with gr.Blocks(theme=gr.themes.Default()) as interface:
    gr.Markdown(
        """
        # Transaction Classifier
        Predict if a transaction is normal or fraudulent. All numbers must be positive.
        """
    )

    with gr.Group():
        distance_from_home = gr.Number(
            label="Distance from Home (km)",
            precision=2,
            minimum=0.01,
            value=0
        )
        distance_from_last_transaction = gr.Number(
            label="Distance from Last (km)",
            precision=2,
            minimum=0.01,
            value=0
        )
        ratio_to_median_purchase_price = gr.Number(
            label="Ratio to Median Price",
            precision=2,
            minimum=0.01,
            value=0
        )
        repeat_retailer = gr.Dropdown(
            choices=["Yes", "No"],
            label="Repeat Retailer",
            value="No"
        )
        used_chip = gr.Dropdown(
            choices=["Yes", "No"],
            label="Used Chip",
            value="Yes"
        )
        used_pin_number = gr.Dropdown(
            choices=["Yes", "No"],
            label="Used PIN",
            value="No"
        )
        online_order = gr.Dropdown(
            choices=["Yes", "No"],
            label="Online Order",
            value="No"
        )

    predict_button = gr.Button("Predict")

    output = gr.Textbox(label="Result", lines=2)

    # Handle prediction
    predict_button.click(
        fn=lambda *args: format_output(predict_transaction(*args)),
        inputs=[
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_retailer,
            used_chip,
            used_pin_number,
            online_order
        ],
        outputs=output
    )

# Launch the interface
interface.launch(share=True)