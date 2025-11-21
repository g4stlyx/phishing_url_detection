import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

class PhishingDataPreprocessor:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        print(f"Loading data from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)

    def clean_data(self):
        """
        Drops irrelevant columns and handles missing values.
        """
        print("Cleaning data...")
        # Drop metadata and high-cardinality text columns
        #! URLSimilarityIndex dropped to prevent data leakage
        cols_to_drop = ['FILENAME', 'URL', 'Domain', 'Title', 'URLSimilarityIndex']
        print(f"Dropping columns: {cols_to_drop}")
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
        # Drop duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Dropped {initial_rows - len(self.df)} duplicate rows.")

    def encode_features(self):
        """
        Encodes categorical features.
        """
        print("Encoding categorical features...")
        # TLD is the main categorical feature remaining
        if 'TLD' in self.df.columns:
            print("Encoding 'TLD' with Label Encoding (Top 20, others 'Other')...")
            # Keep top 20 TLDs, map others to 'Other'
            top_tlds = self.df['TLD'].value_counts().nlargest(20).index
            self.df['TLD'] = self.df['TLD'].apply(lambda x: x if x in top_tlds else 'Other')
            
            le = LabelEncoder()
            self.df['TLD'] = le.fit_transform(self.df['TLD'])
            self.label_encoders['TLD'] = le
            
            # Save encoder
            joblib.dump(le, os.path.join(self.output_dir, 'tld_encoder.joblib'))

    def handle_outliers(self):
        """
        Optional: Handle outliers. 
        For this project, we'll rely on Scaling, but we could clip extreme values.
        """
        print("Handling outliers (Clipping URLLength to 99th percentile)...")
        if 'URLLength' in self.df.columns:
            limit = self.df['URLLength'].quantile(0.99)
            self.df['URLLength'] = self.df['URLLength'].clip(upper=limit)

    def split_and_scale(self):
        """
        Splits data into Train/Test and scales features.
        """
        print("Splitting and Scaling data...")
        target_col = 'label'
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        # Split 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=61, stratify=y)

        # Scale numerical features
        # Identify numerical columns (all remaining X columns should be numeric now)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        print("Fitting StandardScaler on Train data...")
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.output_dir, 'scaler.joblib'))

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        print(f"Saving processed data to {self.output_dir}...")
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        print("Data saved.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    output_dir = os.path.join(base_dir, 'data', 'processed')

    preprocessor = PhishingDataPreprocessor(input_path, output_dir)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.encode_features()
    preprocessor.handle_outliers()
    X_train, X_test, y_train, y_test = preprocessor.split_and_scale()
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
