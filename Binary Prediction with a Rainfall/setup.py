import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shutil

def main():
    print("Setting up Rainfall Prediction Project...")
    
    # Create directories
    directories = ["data", "models", "images"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Check if original data exists
    train_path = "./data/train.csv"
    test_path = "./data/test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Original data files not found. Creating sample data for demonstration...")
        
        # Create sample train data
        np.random.seed(42)
        train_data = pd.DataFrame({
            'id': range(1, 1001),
            'pressure': np.random.uniform(990, 1030, 1000),
            'maxtemp': np.random.uniform(20, 35, 1000),
            'temparature': np.random.uniform(15, 30, 1000),
            'mintemp': np.random.uniform(10, 25, 1000),
            'dewpoint': np.random.uniform(5, 20, 1000),
            'humidity': np.random.uniform(40, 95, 1000),
            'cloud': np.random.randint(0, 9, 1000),
            'sunshine': np.random.uniform(0, 12, 1000),
            'winddirection': np.random.uniform(0, 360, 1000),
            'windspeed': np.random.uniform(0, 50, 1000)
        })
        
        # Create target variable with some correlation to features
        probability = (train_data['cloud'] / 8 * 0.4) + (train_data['humidity'] / 100 * 0.6)
        train_data['rainfall'] = (probability > 0.6).astype(int)
        
        # Create test data
        test_data = pd.DataFrame({
            'id': range(1001, 1301),
            'pressure': np.random.uniform(990, 1030, 300),
            'maxtemp': np.random.uniform(20, 35, 300),
            'temparature': np.random.uniform(15, 30, 300),
            'mintemp': np.random.uniform(10, 25, 300),
            'dewpoint': np.random.uniform(5, 20, 300),
            'humidity': np.random.uniform(40, 95, 300),
            'cloud': np.random.randint(0, 9, 300),
            'sunshine': np.random.uniform(0, 12, 300),
            'winddirection': np.random.uniform(0, 360, 300),
            'windspeed': np.random.uniform(0, 50, 300)
        })
        
        # Save data
        train_data.to_csv(train_path, index=False)
        test_data.to