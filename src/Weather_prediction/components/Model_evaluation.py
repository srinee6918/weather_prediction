import os
import sys
import mlflow
import pickle
import numpy as np
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.Weather_prediction.utils.utils import load_object


class ModelEvaluation:
    def __init__(self):
        pass

    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        return accuracy, precision, recall, f1, roc_auc


    def initiate_model_evaluation(self,train_array,test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:,-1])

            model_path=os.path.join("artifacts","final_model.pkl")

            model=load_object(model_path)

            mlflow.set_registry_uri("http://127.0.0.1:5000")
                        
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            print(tracking_url_type_store)

            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (accuracy, precision, recall, f1, roc_auc) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                print(f"Accuracy: {accuracy}")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print(f"F1 Score: {f1}")
                print(f"ROC AUC: {roc_auc}")


                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "Model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "Model")
            
        except Exception as e:
            raise e