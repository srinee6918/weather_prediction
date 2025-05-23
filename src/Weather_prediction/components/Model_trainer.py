import os
import sys
import pandas as pd
import numpy as np
from src.Weather_prediction.logger import logging
from src.Weather_prediction.exception import customexception
from dataclasses import dataclass
from src.Weather_prediction.utils.utils import save_object, evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

@dataclass 
class ModelTrainerConfig:
    def __init__(self):
        self.artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.trained_model_file_path = os.path.join(self.artifacts_dir, 'final_model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LogisticRegression': LogisticRegression(),
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'XGBClassifier': XGBClassifier()
            }
            
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n' + '='*80 + '\n')
            logging.info(f'Model Report: {model_report}')

            # Get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info(f'Best Model: {best_model_name}, Accuracy Score: {best_model_score}')
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Model pickle file saved')
            
            return best_model_name, best_model_score
            
        except Exception as e:
            logging.error('Error in model training')
            raise customexception(e, sys)