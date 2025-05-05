import os
import sys
import numpy as np
import pandas as pd
from src.Weather_prediction.logger import logging
from src.Weather_prediction.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.raw_data_path = os.path.join("Artifacts","raw_data.csv")
        self.train_data_path = os.path.join("Artifacts","train_data.csv")
        self.test_data_path = os.path.join("Artifacts","test_data.csv")
        self.data_path = os.path.join("Notebook_Experiments","Data","usa_rain_prediction_dataset_2024_2025.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        # Create artifacts directory if it doesn't exist
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            # Read the dataset
            data = pd.read_csv(self.ingestion_config.data_path)
            logging.info("Read the Data from the csv file")

            # Save raw data
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Created the raw data file")

            logging.info("Splitting the data into train and test")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logging.info("Data Splitting is done")

            # Save train and test data
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Created the train and test data files")
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
            
        except Exception as e:
            logging.info("Excpetion occured while ingesting the data")
            raise customexception(e,sys)