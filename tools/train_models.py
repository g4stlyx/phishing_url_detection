import pandas as pd
import os
import joblib
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class PhishingModelTrainer:
    def __init__(self, data_dir, output_dir, models_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = []

    def load_data(self):
        print("Loading processed data...")
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # the last column is target 'label'
        target_col = 'label'
        
        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_test = test_df.drop(columns=[target_col])
        self.y_test = test_df[target_col]
        
        print(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")

    def get_models_and_params(self):
        """
        Defines the models and their hyperparameter grids.
        """
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, random_state=61),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'NaiveBayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier(random_state=61),
                'params': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=61, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'SVM': {
                # Using LinearSVC because standard SVC is O(n^2) and too slow for 200k+ samples
                'model': LinearSVC(random_state=61, dual="auto", max_iter=2000),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2']
                }
            },
            'MLP_ANN': {
                'model': MLPClassifier(random_state=61, max_iter=500, early_stopping=True),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001]
                }
            }
        }
        return models

    def train_and_evaluate(self):
        models_config = self.get_models_and_params()
        
        for name, config in models_config.items():
            print(f"\n{'='*20} Training {name} {'='*20}")
            start_time = time.time()
            
            # RandomizedSearchCV for efficiency
            # n_iter=5 to keep it fast for demonstration, increase for better tuning
            search = RandomizedSearchCV(
                config['model'], 
                config['params'], 
                n_iter=5, 
                cv=3, 
                scoring='f1', 
                n_jobs=-1, 
                random_state=61,
                verbose=1
            )
            
            print(f"Starting hyperparameter tuning for {name}...")
            try:
                search.fit(self.X_train, self.y_train)
                best_model = search.best_estimator_
                print(f"Best Params: {search.best_params_}")
                
                # Evaluate
                y_pred = best_model.predict(self.X_test)
                
                # Metrics
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred)
                rec = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                
                # ROC AUC (needs probability or decision function)
                try:
                    if hasattr(best_model, "predict_proba"):
                        y_prob = best_model.predict_proba(self.X_test)[:, 1]
                    else:
                        y_prob = best_model.decision_function(self.X_test)
                    roc = roc_auc_score(self.y_test, y_prob)
                except Exception as e:
                    print(f"Could not calculate ROC AUC for {name}: {e}")
                    roc = 0.0

                elapsed = time.time() - start_time
                print(f"Done in {elapsed:.2f}s. Accuracy: {acc:.4f}, F1: {f1:.4f}")
                
                # Save results
                self.results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1_Score': f1,
                    'ROC_AUC': roc,
                    'Best_Params': str(search.best_params_),
                    'Training_Time': elapsed
                })
                
                # Save model
                model_path = os.path.join(self.models_dir, f"{name}_best.joblib")
                joblib.dump(best_model, model_path)
                
            except Exception as e:
                print(f"Error training {name}: {e}")

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.output_dir, 'model_comparison.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        print(results_df[['Model', 'Accuracy', 'F1_Score', 'ROC_AUC', 'Training_Time']])

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'outputs', 'results')
    models_dir = os.path.join(base_dir, 'outputs', 'models')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    trainer = PhishingModelTrainer(data_dir, output_dir, models_dir)
    trainer.load_data()
    trainer.train_and_evaluate()
    trainer.save_results()

if __name__ == "__main__":
    main()
