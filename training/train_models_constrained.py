import pandas as pd
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ConstrainedPhishingTrainer:
    def __init__(self, data_dir, output_dir, models_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = []

    def load_and_clean_data(self):
        print("Loading processed data...")
        # Load original processed data
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        
        # Combine to re-split
        full_df = pd.concat([train_df, test_df])
        
        # Features to drop
        drop_features = [
            'URLSimilarityIndex', 'HasSocialNet', 'HasCopyrightInfo', 'HasDescription', 
            'SpacialCharRatioInURL', 'IsHTTPS', 'DomainTitleMatchScore', 'HasSubmitButton', 
            'IsResponsive', 'URLTitleMatchScore', 'NoOfExternalRef', 'NoOfJS', 'LineOfCode'
        ]
        print(f"Dropping features to reduce overfitting: {drop_features}")
        
        # Check if columns exist before dropping to avoid errors
        existing_cols = [col for col in drop_features if col in full_df.columns]
        full_df = full_df.drop(columns=existing_cols)
        
        target_col = 'label'
        X = full_df.drop(columns=[target_col])
        y = full_df[target_col]
        
        # Split again
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=61, stratify=y
        )
        
        print(f"New Train shape: {self.X_train.shape}")

    def train_models(self):
        # Constrained models to prevent overfitting
        models = {
            'DecisionTree': DecisionTreeClassifier(
                random_state=61, 
                max_depth=3,              # Reduced depth
                min_samples_split=50      # Require more samples to split
            ),
            'RandomForest': RandomForestClassifier(
                random_state=61, 
                n_estimators=10,          # Fewer trees
                max_depth=3,              # Reduced depth
                min_samples_split=50,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=61, 
                max_iter=50,               # Reduced iterations
                C=0.01                     # Stronger regularization (smaller C)
            ),
            'NaiveBayes': GaussianNB(),    # Simple model, hard to constrain further
            'SVM': LinearSVC(
                random_state=61, 
                dual="auto", 
                max_iter=50,              # Reduced iterations
                C=0.01                    # Stronger regularization
            ),
            'MLP_ANN': MLPClassifier(
                random_state=61, 
                max_iter=50,              # Reduced iterations
                hidden_layer_sizes=(20,), # Very small single hidden layer
                early_stopping=True,
                alpha=0.01                # Higher regularization
            )
        }
        
        print("\nTraining constrained models...")
        
        for name, model in models.items():
            start = time.time()
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred)
                rec = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                
                # ROC AUC
                try:
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(self.X_test)[:, 1]
                    else:
                        y_prob = model.decision_function(self.X_test)
                    roc = roc_auc_score(self.y_test, y_prob)
                except Exception as e:
                    print(f"  Could not calculate ROC AUC: {e}")
                    roc = 0.0
                
                elapsed = time.time() - start
                print(f"\n{name}:")
                print(f"  Accuracy: {acc:.4f}")
                print(f"  Precision: {prec:.4f}")
                print(f"  Recall: {rec:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  ROC AUC: {roc:.4f}")
                print(f"  Time: {elapsed:.2f}s")
                
                self.results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1_Score': f1,
                    'ROC_AUC': roc,
                    'Training_Time': elapsed
                })

                # Print Feature Importance for Tree models
                if hasattr(model, 'feature_importances_'):
                    importances = pd.Series(model.feature_importances_, index=self.X_train.columns)
                    print(f"  Top 5 Features:\n{importances.sort_values(ascending=False).head(5)}")
                
                # Save model
                model_path = os.path.join(self.models_dir, f"{name}_constrained.joblib")
                joblib.dump(model, model_path)
                print(f"  Model saved to: {model_path}")
                    
            except Exception as e:
                print(f"Error training {name}: {e}")

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.output_dir, 'constrained_model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print('='*60)
        print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'Training_Time']])
        print(f"\nDetailed results saved to: {results_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'outputs', 'results_after_simplifying')
    models_dir = os.path.join(base_dir, 'outputs', 'results_after_simplifying', 'models')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    trainer = ConstrainedPhishingTrainer(data_dir, output_dir, models_dir)
    trainer.load_and_clean_data()
    trainer.train_models()
    trainer.save_results()
    trainer.save_results()

if __name__ == "__main__":
    main()
