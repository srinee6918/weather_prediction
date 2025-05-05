import os
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load model and preprocessors
model = None
scaler = None
encoder = None

def load_artifacts():
    global model, scaler, encoder
    try:
        model = load('artifacts/model.joblib')
        scaler = load('artifacts/scaler.joblib')
        encoder = load('artifacts/encoder.joblib')
        return True
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        return False

def preprocess_input(data):
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # One-hot encode categorical columns
        cat_cols = ['Location']
        num_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure']
        
        # Scale numerical features
        scaled_numerical = scaler.transform(input_df[num_cols])
        scaled_df = pd.DataFrame(scaled_numerical, columns=num_cols)
        
        # Encode categorical features
        one_hot_encoded = encoder.transform(input_df[cat_cols])
        one_hot_df = pd.DataFrame(one_hot_encoded, 
                                columns=encoder.get_feature_names_out(cat_cols))
        
        # Combine features
        final_input = pd.concat([scaled_df, one_hot_df], axis=1)
        return final_input
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not all([model, scaler, encoder]):
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get form data
        data = {
            'Location': request.form.get('Location'),
            'Temperature': float(request.form.get('Temperature')),
            'Humidity': float(request.form.get('Humidity')),
            'Wind Speed': float(request.form.get('Wind_Speed')),
            'Precipitation': float(request.form.get('Precipitation')),
            'Cloud Cover': float(request.form.get('Cloud_Cover')),
            'Pressure': 1013.25  # Default pressure if not provided
        }
        
        # Preprocess input
        processed_input = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
        
        result = {
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'probability': float(probability)
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    if load_artifacts():
        app.run(debug=True)
    else:
        print("Failed to load model artifacts. Please train the model first.")