import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class RobustPhishingTrainer:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.results = []

    def load_and_clean_data(self):
        print("Loading processed data...")
        # Load original processed data
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        # Combine to re-split after dropping feature
        full_df = pd.concat([train_df, test_df])
        
        # DROP SUSPICIOUS FEATURE
        print("Dropping 'URLSimilarityIndex' due to potential data leakage...")
        full_df = full_df.drop(columns=['URLSimilarityIndex'])
        
        target_col = 'label'
        X = full_df.drop(columns=[target_col])
        y = full_df[target_col]
        
        # Split again
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=61, stratify=y
        )
        
        print(f"New Train shape: {self.X_train.shape}")

    def train_models(self):
        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=61, max_depth=20),
            'RandomForest': RandomForestClassifier(random_state=61, n_estimators=50, n_jobs=-1),
            'LogisticRegression': LogisticRegression(random_state=61, max_iter=1000),
            'NaiveBayes': GaussianNB()
        }
        
        print("\nRetraining models without 'URLSimilarityIndex'...")
        
        for name, model in models.items():
            start = time.time()
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            elapsed = time.time() - start
            print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f} ({elapsed:.2f}s)")
            
            self.results.append({
                'Model': name,
                'Accuracy': acc,
                'F1_Score': f1
            })

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'outputs', 'results')
    
    trainer = RobustPhishingTrainer(data_dir, output_dir)
    trainer.load_and_clean_data()
    trainer.train_models()

if __name__ == "__main__":
    main()
