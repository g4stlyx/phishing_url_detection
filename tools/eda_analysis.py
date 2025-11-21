import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set plot style
sns.set(style="whitegrid")

class PhishingDataAnalyzer:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """Loads the dataset from the CSV file."""
        print(f"Loading data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)

    def describe_dataset(self):
        """
        Section 2: Dataset Description
        - Source, Size, Types, Target Variable
        """
        print("\n" + "="*50)
        print("2. DATASET DESCRIPTION")
        print("="*50)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Number of Samples: {self.df.shape[0]}")
        print(f"Number of Features: {self.df.shape[1]}")
        
        print("\nColumn Data Types:")
        print(self.df.dtypes.value_counts())
        
        # Identify target variable (assuming 'label' or 'Label')
        target_col = [col for col in self.df.columns if 'label' in col.lower()]
        if target_col:
            self.target_col = target_col[0]
            print(f"\nTarget Variable: '{self.target_col}'")
            print(self.df[self.target_col].value_counts(normalize=True))
        else:
            print("\nWARNING: Target variable not found automatically.")
            self.target_col = None

        print("\nFirst 5 rows:")
        print(self.df.head())

    def analyze_preprocessing_needs(self):
        """
        Section 3: Data Preprocessing Analysis
        - Missing values
        - Outliers
        - Duplicates
        - Class Balance
        """
        print("\n" + "="*50)
        print("3. PREPROCESSING ANALYSIS")
        print("="*50)

        # Missing Values
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("Missing Values: None detected.")
        else:
            print("Missing Values detected:")
            print(missing)

        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate Rows: {duplicates}")

        # Class Balance
        if self.target_col:
            class_counts = self.df[self.target_col].value_counts()
            print(f"\nClass Balance:\n{class_counts}")
            is_balanced = (class_counts.min() / class_counts.max()) > 0.8
            print(f"Is Balanced? {'Yes' if is_balanced else 'No (Consider SMOTE or resampling)'}")

        # Outliers (using IQR for numerical columns)
        print("\nOutlier Detection (Sample of numerical features):")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Exclude target
        if self.target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target_col)
        
        for col in numeric_cols[:5]: # Check first 5 numerical columns for brevity
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            print(f" - {col}: {outliers} outliers detected (IQR method)")

    def perform_eda(self):
        """
        Section 4: Exploratory Data Analysis (EDA)
        - Summary statistics
        - Visualizations
        """
        print("\n" + "="*50)
        print("4. EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*50)

        # Summary Statistics
        print("\nSummary Statistics:")
        print(self.df.describe())

        # Visualizations
        print(f"\nGenerating plots in {self.output_dir}...")

        # 1. Target Distribution
        if self.target_col:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=self.target_col, data=self.df)
            plt.title('Distribution of Target Variable')
            plt.savefig(os.path.join(self.output_dir, 'target_distribution.png'))
            plt.close()
            print(" - Saved target_distribution.png")

        # 2. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        # Select only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        sns.heatmap(corr, cmap='coolwarm', annot=False) # annot=False because 54 features is too crowded
        plt.title('Feature Correlation Heatmap')
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()
        print(" - Saved correlation_heatmap.png")

        # 3. Boxplots for top correlated features with target
        if self.target_col:
            # Find features most correlated with target
            target_corr = corr[self.target_col].abs().sort_values(ascending=False)
            top_features = target_corr.index[1:6] # Top 5 excluding target itself
            
            for feature in top_features:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=self.target_col, y=feature, data=self.df)
                plt.title(f'{feature} vs Target')
                plt.savefig(os.path.join(self.output_dir, f'boxplot_{feature}.png'))
                plt.close()
                print(f" - Saved boxplot_{feature}.png")

        # 4. Histograms for some features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.df, x=col, hue=self.target_col, kde=True, element="step")
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(self.output_dir, f'hist_{col}.png'))
            plt.close()
            print(f" - Saved hist_{col}.png")

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    output_dir = os.path.join(base_dir, 'outputs', 'eda')

    analyzer = PhishingDataAnalyzer(data_path, output_dir)
    analyzer.load_data()
    analyzer.describe_dataset()
    analyzer.analyze_preprocessing_needs()
    analyzer.perform_eda()
    
    print("\nAnalysis Complete. Check the outputs directory for plots.")

if __name__ == "__main__":
    main()
