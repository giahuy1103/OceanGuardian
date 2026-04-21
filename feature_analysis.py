import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance():
    print("Starting feature importance analysis module...")
    
    train_path = 'data/processed/train_split.csv'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run data_builder.py first!")
        return

    # 1. Load safely processed data
    train_df = pd.read_csv(train_path)
    
    # 2. Prepare Tabular features
    features = ['SST (°C)', 'pH Level', 'Latitude', 'Longitude', 'Month', 'Marine Heatwave']
    X_train = train_df[features]
    y_train = train_df['Label']
    
    # 3. Initialize Random Forest algorithm
    # Increase n_estimators and limit max_depth to make trees more stable on small datasets
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 4. Print results to Terminal
    print("\n=== ENVIRONMENTAL FEATURE IMPORTANCE RANKING ===")
    for f in range(X_train.shape[1]):
        # Align for better readability
        print(f"{f + 1}. {features[indices[f]]:<18} ({importances[indices[f]]:.4f})")
        
    # 5. Visualize with charts
    try:
        plt.figure(figsize=(8, 5))
        
        # Plot Barplot with scientific color palette (viridis)
        sns.barplot(
            x=[importances[i] for i in indices], 
            y=[features[i] for i in indices], 
            palette="viridis",
            hue=[features[i] for i in indices],
            legend=False
        )
        
        plt.title("Environmental Feature Importance (Random Forest)", fontsize=14, fontweight='bold')
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        
        # Add faint grid lines on the x-axis for easier reading
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the chart
        os.makedirs('data/processed', exist_ok=True)
        plot_path = 'data/processed/feature_importance.png'
        plt.savefig(plot_path, dpi=300)
        print(f"\nFeature Importance chart exported to: {plot_path}")
        
    except Exception as e:
        print(f"\nCannot draw the chart: {e}. (Hint: Install matplotlib and seaborn libraries)")

if __name__ == '__main__':
    analyze_feature_importance()