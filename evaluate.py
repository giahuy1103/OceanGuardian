import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, confusion_matrix

# Import charting libraries
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import CoralMultimodalDataset
from src.model import CoralFusionModel

def evaluate_model():
    print("Initializing OceanGuardian Evaluation system...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. LOAD CHECKPOINT (Including Weights and env_stats)
    checkpoint_path = 'data/processed/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please run train.py first!")
        return

    print("Extracting weights and environmental parameters from Checkpoint...")
    # Note: Set weights_only=False to allow loading the dictionary containing env_stats
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract normalization parameters (Crucial to avoid Data Leakage)
    env_stats = checkpoint['env_stats']
    
    # Initialize and load weights for the model
    model = CoralFusionModel(num_classes=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. LOAD VALIDATION DATASET
    val_path = 'data/processed/val_split.csv'
    val_df = pd.read_csv(val_path)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # INITIALIZE DATASET WITH env_stats SYNCHRONIZED FROM TRAIN SET
    val_dataset = CoralMultimodalDataset(val_df, transform=transform, env_stats=env_stats)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    all_labels = []
    all_probs = []

    # 3. INFERENCE PROCESS
    print("Performing inference on the Validation set...")
    with torch.no_grad():
        for images, env_features, labels in val_loader:
            images, env_features, labels = images.to(device), env_features.to(device), labels.to(device)
            outputs = model(images, env_features)
            
            # Use Softmax to get the predicted probability for label 1 (Bleached)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 4. DECISION THRESHOLD CALIBRATION
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    # Calculate F1-score for all possible thresholds
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    
    # Get the threshold with the highest F1-score
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]

    print(f"\n=== THRESHOLD CALIBRATION RESULTS ===")
    print(f"Default threshold (0.500) -> F1-score: {f1_score(all_labels, (all_probs >= 0.5).astype(int)):.4f}")
    print(f"Optimal threshold ({optimal_threshold:.4f}) -> F1-score: {optimal_f1:.4f}")

    # 5. FINAL EVALUATION WITH OPTIMAL THRESHOLD
    final_preds = (all_probs >= optimal_threshold).astype(int)
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, final_preds, target_names=['Healthy (0)', 'Bleached (1)']))

    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(all_labels, final_preds)
    print(cm)
    
    # 6. VISUALIZATION (EXPORT IMAGE FILE)
    try:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Healthy (0)', 'Bleached (1)'], 
                    yticklabels=['Healthy (0)', 'Bleached (1)'],
                    annot_kws={"size": 14, "weight": "bold"})
        
        plt.title(f"Confusion Matrix (Threshold: {optimal_threshold:.2f})", fontsize=14)
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        
        img_path = 'data/processed/confusion_matrix.png'
        plt.savefig(img_path, dpi=300)
        print(f"\nConfusion Matrix chart exported to: {img_path}")
    except Exception as e:
        print(f"\nCannot draw the chart: {e}. (Hint: Install matplotlib and seaborn libraries)")

if __name__ == '__main__':
    evaluate_model()