import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Weather_prediction.exception import customexception
from src.Weather_prediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.Weather_prediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.artifacts_dir = os.path.join(os.getcwd(), 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.preprocessor_obj_file_path = os.path.join(self.artifacts_dir, 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

        
    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')
            
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Location']
            numerical_cols = ['Temperature','Humidity','Wind Speed','Precipitation','Cloud Cover']
            
           
            
            logging.info('Pipeline Initiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]

            )
            
            preprocessor=ColumnTransformer([
            ('num',num_pipeline,numerical_cols),
            ('cat',cat_pipeline,categorical_cols)
            ])
            
            logging.info('Pipeline Completed')
            return preprocessor
            

            
            
        
        except Exception as e:
            logging.error('Error in get_data_transformation')
            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()
            
            target_column_name = 'Rain Tomorrow'
            drop_columns = [target_column_name,'Date']
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            
            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.error('Error in initialize_data_transformation')
            raise customexception(e,sys)
            
    
