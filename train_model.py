import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
import matplotlib.pyplot as plt

def train_weather_model():
    try:
        # Load the dataset
        data_path = r'Notebook_Experiments\Data\usa_rain_prediction_dataset_2024_2025.csv'
        df = pd.read_csv(data_path)
        
        # Drop Date column
        df.drop('Date', axis=1, inplace=True)
        
        # Separate categorical and numerical columns
        cat_cols = ['Location']
        num_cols = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure']
        
        # One-hot encode categorical columns
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        one_hot_encoded = encoder.fit_transform(df[cat_cols])
        one_hot_df = pd.DataFrame(one_hot_encoded, 
                                columns=encoder.get_feature_names_out(cat_cols))
        
        # Scale numerical columns
        scaler = RobustScaler()
        scaled_numerical = pd.DataFrame(scaler.fit_transform(df[num_cols]), 
                                      columns=num_cols)
        
        # Combine features
        X = pd.concat([scaled_numerical, one_hot_df], axis=1)
        y = df['Rain Tomorrow']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Create artifacts directory if it doesn't exist
        os.makedirs('artifacts', exist_ok=True)
        
        # Save model and preprocessing objects
        dump(model, 'artifacts/model.joblib')
        dump(scaler, 'artifacts/scaler.joblib')
        dump(encoder, 'artifacts/encoder.joblib')
        
        # Evaluate model
        y_pred = model.predict(X_test)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        print(f"Model trained successfully with F2 Score: {f2:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Confusion Matrix')
        plt.savefig('artifacts/confusion_matrix.png')
        plt.close()
        
        print("Training completed. Model and preprocessors saved in 'artifacts' directory.")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_weather_model()