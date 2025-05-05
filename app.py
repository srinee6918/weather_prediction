import os
import sys
from flask import Flask, request, render_template, jsonify
import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.Weather_prediction.pipelines.Prediction_pipeline import PredictPipeline, WeatherPredictionInputFeatureData
    from src.Weather_prediction.logger import logging
    from src.Weather_prediction.exception import customexception
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure you've renamed 'Weather prediction' directory to 'weather_prediction'")
    raise

application = Flask(__name__)
app = application

# Route for the home page (displaying the form)
@app.route('/')
def index():
    return render_template('form.html')

# Route for handling the prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    try:
        logging.info("Received form data:")
        form_data = request.form
        for key, value in form_data.items():
            logging.info(f"{key}: {value}")

        # Validate required fields
        required_fields = ['Location', 'Temperature', 'Humidity', 'Wind_Speed', 'Precipitation', 'Cloud_Cover']
        missing_fields = [field for field in required_fields if not form_data.get(field)]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Create the input data object from form data
        try:
            data = WeatherPredictionInputFeatureData(
                Location=form_data.get('Location'),
                Temperature=float(form_data.get('Temperature')),
                Humidity=float(form_data.get('Humidity')),
                Wind_Speed=float(form_data.get('Wind_Speed')),
                Precipitation=float(form_data.get('Precipitation')),
                Cloud_Cover=form_data.get('Cloud_Cover')
            )
        except ValueError as ve:
            raise ValueError(f"Invalid input data: {str(ve)}")

        # Log the input data
        logging.info("Input data validated and created")

        # Get the data as a DataFrame
        pred_df = data.get_data_as_dataframe()
        logging.info("Input DataFrame created:")
        logging.info(f"\n{pred_df.to_string()}")

        # Instantiate the prediction pipeline and predict
        logging.info("Initializing PredictPipeline...")
        predict_pipeline = PredictPipeline()
        logging.info("Making prediction...")
        
        try:
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Raw prediction results: {results}")
            prediction_result = results[0] if len(results) > 0 else "No prediction returned"
            logging.info(f"Final prediction result: {prediction_result}")
            print("Final prediction result:", prediction_result)
            
            # Return the result
            return render_template('result.html', final_result=prediction_result)
            
        except Exception as pred_error:
            logging.error(f"Prediction error: {str(pred_error)}", exc_info=True)
            error_msg = f"Prediction failed: {str(pred_error)}"
            if hasattr(pred_error, '__traceback__'):
                import traceback
                error_msg += f"\n\nTraceback:\n{''.join(traceback.format_tb(pred_error.__traceback__))}"
            raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        logging.error(f"Error in predict_datapoint: {error_msg}", exc_info=True)
        
        # For AJAX requests, return JSON response
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.headers.get('Content-Type') == 'application/x-www-form-urlencoded':
            return jsonify({
                "error": "Failed to process prediction",
                "details": str(e),
                "type": type(e).__name__
            }), 400
        else:
            return render_template('form.html', error=error_msg)


if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("Starting Flask development server...")
        print(f" * Running on http://127.0.0.1:5000/")
        print(f" * Running on http://localhost:5000/")
        print("="*50 + "\n")
        
        # Run the Flask app with debug mode on
        app.run(host="0.0.0.0", port=5000, debug=True)
        
    except Exception as e:
        print("\n" + "!"*50)
        print(f"Error starting the server: {str(e)}")
        print("!"*50 + "\n")
