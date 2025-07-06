import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(path):
    # Load parameters from a YAML file
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f"Parameters loaded from {path}")
        logger.debug(f"Parameters: {params}")
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(path):
    # Load data from a CSV file
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded from {path}")
        logger.debug(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess(df):
    # Preprocess the DataFrame
    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['clean_comment'].str.strip() != '']
        logger.info("Data preprocessing completed")
        logger.debug(f"Processed data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def save_data(train, test, path):
    try:
        raw_path = os.path.join(path, 'raw')
        os.makedirs(raw_path, exist_ok=True)

        train_path = os.path.join(raw_path, 'train.csv')
        test_path = os.path.join(raw_path, 'test.csv')

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        logger.info(f"Train and test data saved to {raw_path}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def main():
    try:
        print("Loading params...")
        params = load_params(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'))
        test_size = params['ingest']['test_size']
        
        print("Loading data...")
        df = load_data('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        
        print("Preprocessing...")
        final_df = preprocess(df)
        
        print("Splitting...")
        train, test = train_test_split(final_df, test_size=test_size, random_state=42)
        
        print("Saving...")
        save_data(train, test, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

        print("Done ✅")
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
