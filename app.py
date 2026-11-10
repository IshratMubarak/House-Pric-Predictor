
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the saved model pipeline once when the app starts
MODEL_PATH = os.path.join('model', 'best_house_price_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# Exact features expected by your trained model pipeline
NUM_FEATURES = [
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    'TotalBsmtSF',
    'GrLivArea'
]

CAT_FEATURES = [
    'MSSubClass',
    'MSZoning',
    'BldgType',
    'Neighborhood'
]


@app.route('/')
def home():
    return render_template('index.html', error=None)


@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return render_template(
            'index.html',
            error="Model file not found. Please place "
                  "'best_house_price_model.pkl' inside the 'model/' folder."
        )

    try:
        # Collect and validate numerical features
        input_data = {}

        for col in NUM_FEATURES:
            val = request.form.get(col)

            if val is None or val.strip() == "":
                raise ValueError(
                    f"Missing required numeric field: {col}"
                )

            input_data[col] = float(val)

        # Collect and validate categorical features
        for col in CAT_FEATURES:
            val = request.form.get(col)

            if val is None or val.strip() == "":
                raise ValueError(
                    f"Missing required categorical field: {col}"
                )

            val = val.strip()

            # MSSubClass was integer-coded when the model was trained
            input_data[col] = (
                int(val) if col == "MSSubClass" else val
            )

        # Create DataFrame with exact column structure and order
        df_input = pd.DataFrame([input_data])

        # Predict using the saved pipeline
        # (handles preprocessing automatically)
        prediction = model.predict(df_input)[0]

        formatted_price = f"${prediction:,.2f}"

        return render_template(
            'result.html',
            prediction=formatted_price,
            user_inputs=input_data
        )

    except ValueError as ve:
        return render_template(
            'index.html',
            error=f"Input Error: {str(ve)}"
        )

    except Exception as e:
        print(f"Prediction error: {e}")

        return render_template(
            'index.html',
            error="An unexpected error occurred during prediction. "
                  "Please check your inputs."
        )


if __name__ == '__main__':
    app.run(debug=True)
