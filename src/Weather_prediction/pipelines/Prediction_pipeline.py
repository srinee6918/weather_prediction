import os
import sys
import pandas as pd
from src.Weather_prediction.exception import customexception
from src.Weather_prediction.logger import logging
from src.Weather_prediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        # Create artifacts directory if it doesn't exist
        self.artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
    
    def predict(self, features):
        try:
            # Use consistent path with os.path.join
            preprocessor_path = os.path.join(self.artifacts_dir, 'preprocessor.pkl')
            model_path = os.path.join(self.artifacts_dir, 'final_model.pkl')
            
            # Check if files exist
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            logging.info(f"Loading preprocessor from {preprocessor_path}")
            preprocessor = load_object(preprocessor_path)
            logging.info(f"Loading model from {model_path}")
            model = load_object(model_path)
            
            logging.info("Transforming input features")
            scaled_data = preprocessor.transform(features)
            
            logging.info("Making prediction")
            pred = model.predict(scaled_data)
            
            return pred
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise customexception(e, sys)
    
    
class WeatherPredictionInputFeatureData:
    def __init__(self,
                 Location: str,
                 Temperature: float,
                 Humidity: float,
                 Wind_Speed: float,
                 Precipitation: float,
                 Cloud_Cover: str):  # Changed type hint to str since it's a string
        
        self.Location = Location
        self.Temperature = Temperature
        self.Humidity = Humidity
        self.Wind_Speed = Wind_Speed
        self.Precipitation = Precipitation
        self.Cloud_Cover = Cloud_Cover
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Location': [self.Location],
                'Temperature': [self.Temperature],
                'Humidity': [self.Humidity],
                'Wind Speed': [self.Wind_Speed],
                'Precipitation': [self.Precipitation],
                'Cloud Cover': [self.Cloud_Cover]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
            
        except Exception as e:
            logging.error('Exception occurred in get_data_as_dataframe')
            raise customexception(e, sys)
