import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_table_image():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'outputs', 'results', 'model_comparison.csv')
    output_path = os.path.join(base_dir, 'outputs', 'results', 'model_comparison_table.png')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Select relevant columns and round
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    df_display = df[cols].copy()
    for col in cols[1:]:
        df_display[col] = df_display[col].round(4)
    
    # Sort by F1 Score
    df_display = df_display.sort_values(by='F1_Score', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4)) # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df_display.values,
                     colLabels=df_display.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) # Scale width, height
    
    # Style headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
    
    plt.title('Model Performance Comparison', pad=20, fontsize=14, weight='bold')
    plt.tight_layout()
    
    print(f"Saving table image to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Done.")

if __name__ == "__main__":
    generate_table_image()
