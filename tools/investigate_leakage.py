import pandas as pd
import numpy as np
import os

def investigate_leakage():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check correlation with label
    print("\nCalculating correlations with 'label'...")
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['label'].sort_values(ascending=False)
    
    print("\nTop 10 Positive Correlations:")
    print(correlations.head(10))
    
    print("\nTop 10 Negative Correlations:")
    print(correlations.tail(10))
    
    # Investigate URLSimilarityIndex specifically
    print("\nInvestigating 'URLSimilarityIndex'...")
    print(df.groupby('label')['URLSimilarityIndex'].describe())
    
    # Check if URLSimilarityIndex perfectly separates the classes
    # Let's see if there is a threshold that separates them perfectly
    phishing = df[df['label'] == 1]['URLSimilarityIndex']
    legit = df[df['label'] == 0]['URLSimilarityIndex']
    
    print(f"\nPhishing URLSimilarityIndex - Min: {phishing.min()}, Max: {phishing.max()}")
    print(f"Legitimate URLSimilarityIndex - Min: {legit.min()}, Max: {legit.max()}")
    
    # Check overlap
    overlap_min = max(phishing.min(), legit.min())
    overlap_max = min(phishing.max(), legit.max())
    
    if overlap_min < overlap_max:
        print(f"Overlap range: {overlap_min} to {overlap_max}")
        overlap_count = df[(df['URLSimilarityIndex'] >= overlap_min) & (df['URLSimilarityIndex'] <= overlap_max)].shape[0]
        print(f"Number of samples in overlap region: {overlap_count}")
    else:
        print("No overlap in range!")

    # Check other high correlation features
    # LineOfCode
    print("\nInvestigating 'LineOfCode'...")
    print(df.groupby('label')['LineOfCode'].describe())

if __name__ == "__main__":
    investigate_leakage()
