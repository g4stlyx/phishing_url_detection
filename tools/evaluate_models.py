import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix

class ModelEvaluator:
    def __init__(self, data_dir, models_dir, output_dir):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.X_test = None
        self.y_test = None
        self.models = {}
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        print("Loading test data...")
        test_path = os.path.join(self.data_dir, 'test.csv')
        test_df = pd.read_csv(test_path)
        self.y_test = test_df['label']
        self.X_test = test_df.drop(columns=['label'])

    def load_models(self):
        print("Loading trained models...")
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
        for f in model_files:
            model_name = f.replace('_best.joblib', '')
            self.models[model_name] = joblib.load(os.path.join(self.models_dir, f))
        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def _get_X_for_model(self, model):
        if hasattr(model, 'feature_names_in_'):
            return self.X_test[model.feature_names_in_]
        return self.X_test

    def plot_roc_curves(self):
        print("Generating ROC Curves...")
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            try:
                X_subset = self._get_X_for_model(model)
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_subset)[:, 1]
                else:
                    y_score = model.decision_function(X_subset)
                
                fpr, tpr, _ = roc_curve(self.y_test, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
            except Exception as e:
                print(f"Skipping ROC for {name}: {e}")

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, 'roc_curves_comparison.png'))
        plt.close()
        print("Saved roc_curves_comparison.png")

    def plot_confusion_matrices(self):
        print("Generating Confusion Matrices...")
        for name, model in self.models.items():
            X_subset = self._get_X_for_model(model)
            y_pred = model.predict(X_subset)
            cm = confusion_matrix(self.y_test, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{name}.png'))
            plt.close()
            print(f"Saved confusion_matrix_{name}.png")

    def plot_feature_importance(self):
        print("Generating Feature Importance Plot...")
        # Prefer RandomForest, then DecisionTree
        model = None
        for name in self.models:
            if 'RandomForest' in name:
                model = self.models[name]
                break
        if not model:
            for name in self.models:
                if 'DecisionTree' in name:
                    model = self.models[name]
                    break
        
        if model:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = self.X_test.columns
                
                feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                feature_imp = feature_imp.sort_values(by='Importance', ascending=False).head(15)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
                plt.title('Top 15 Most Important Features')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
                plt.close()
                print("Saved feature_importance.png")
                print("\nTop 10 Features:")
                print(feature_imp.head(10))
            else:
                print("Selected model does not support feature importance.")
        else:
            print("No tree-based model found for feature importance.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'outputs', 'results_after_simplifying', 'models')
    output_dir = os.path.join(base_dir, 'outputs', 'results_after_simplifying', 'evaluation_plots')

    evaluator = ModelEvaluator(data_dir, models_dir, output_dir)
    evaluator.load_data()
    evaluator.load_models()
    evaluator.plot_roc_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_feature_importance()

if __name__ == "__main__":
    main()
