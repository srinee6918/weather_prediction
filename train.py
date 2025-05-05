import os
import sys
import logging
from src.Weather_prediction.logger import logging
from src.Weather_prediction.exception import customexception
from src.Weather_prediction.components.Data_ingestion import DataIngestion
from src.Weather_prediction.components.Data_transformation import DataTransformation
from src.Weather_prediction.components.Model_trainer import ModelTrainer

def train_pipeline():
    try:
        logging.info("Starting training pipeline")
        
        # 1. Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Train data path: {train_data_path}")
        logging.info(f"Test data path: {test_data_path}")
        
        # 2. Data Transformation
        logging.info("Starting data transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initialize_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )
        logging.info("Data transformation completed")
        
        # 3. Model Training
        logging.info("Starting model training")
        model_trainer = ModelTrainer()
        best_model_name, best_model_score = model_trainer.initate_model_training(
            train_array=train_arr,
            test_array=test_arr
        )
        logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
        
        logging.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise customexception(e, sys)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )
    
    train_pipeline()