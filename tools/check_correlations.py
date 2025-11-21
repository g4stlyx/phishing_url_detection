import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'processed', 'train.csv')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Calculate correlation with label
    target_col = 'label'
    if target_col not in df.columns:
        print("Label column not found!")
        return

    correlations = df.corr()[target_col].sort_values(ascending=False)
    
    print("\nTop 10 Positive Correlations:")
    print(correlations.head(10))

    print("\nTop 10 Negative Correlations:")
    print(correlations.tail(10))

    # Also check for URLSimilarityIndex if it exists
    if 'URLSimilarityIndex' in df.columns:
        print(f"\nURLSimilarityIndex correlation: {correlations.get('URLSimilarityIndex')}")

if __name__ == "__main__":
    main()
