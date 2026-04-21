import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os

# Import modules from the src directory
from src.dataset import CoralMultimodalDataset
from src.model import CoralFusionModel

def calculate_env_stats(train_df):
    """Calculate Mean/Std ONLY on the Train set to avoid Data Leakage."""
    return {
        'sst_mean': train_df['SST (°C)'].mean(),
        'sst_std': train_df['SST (°C)'].std(),
        'ph_mean': train_df['pH Level'].mean(),
        'ph_std': train_df['pH Level'].std(),
        'lat_mean': train_df['Latitude'].mean(),
        'lat_std': train_df['Latitude'].std(),
        'lon_mean': train_df['Longitude'].mean(),
        'lon_std': train_df['Longitude'].std(),
        'month_mean': train_df['Month'].mean(),
        'month_std': train_df['Month'].std(),
    }

def main():
    print("Starting OceanGuardian System (Advanced Research Mode - AdamW & Scheduler Optimization)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 1. Read data directly
    train_path = 'data/processed/train_split.csv'
    val_path = 'data/processed/val_split.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: Train/Val sets not found. Please run data_builder.py first!")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    env_stats = calculate_env_stats(train_df)
    
    # 2. Configure Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Initialize Dataset & Dataloader
    train_dataset = CoralMultimodalDataset(train_df, transform=train_transform, env_stats=env_stats)
    val_dataset = CoralMultimodalDataset(val_df, transform=val_transform, env_stats=env_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 4. Initialize Model, Loss Function, and Optimizer
    model = CoralFusionModel(num_classes=2).to(device)
    
    # Handle class imbalance
    class_counts = train_df['Label'].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = torch.tensor([total_samples / c for c in class_counts], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # OPTIMIZATION 1: Use AdamW instead of Adam, increase weight_decay to 1e-3 for stronger penalty
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    # OPTIMIZATION 2: Add Learning Rate Scheduler
    # If Val Loss does not decrease for 3 epochs, reduce Learning Rate by half (factor=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 5. Configure Early Stopping & Checkpointing
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 10  # OPTIMIZATION 3: Increase patience to 10 to give the scheduler a chance to drop LR
    trigger_times = 0
    save_path = 'data/processed/best_model.pth'
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # --- PHASE 1: TRAINING ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, env_features, labels in train_loader:
            images, env_features, labels = images.to(device), env_features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, env_features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total

        # --- PHASE 2: VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_images, val_env, val_labels in val_loader:
                val_images, val_env, val_labels = val_images.to(device), val_env.to(device), val_labels.to(device)
                val_outputs = model(val_images, val_env)
                v_loss = criterion(val_outputs, val_labels)
                val_running_loss += v_loss.item()
                
        epoch_val_loss = val_running_loss / len(val_loader)
        
        # Scheduler step based on Val Loss
        scheduler.step(epoch_val_loss)
        
        # Get current Learning Rate to print to screen
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] - LR: {current_lr:.6f} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}")

        # --- PHASE 3: CHECKPOINTING & EARLY STOPPING ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            trigger_times = 0
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'env_stats': env_stats
            }, save_path)
            
            print(f"  -> Checkpoint saved! (Val Loss decreased to {best_val_loss:.4f})")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"\n[Early Stopping] Model has stopped converging. Stopping early at Epoch {epoch+1}!")
                break

    print(f"\nTraining complete. Best weights saved at: {save_path}")

if __name__ == '__main__':
    main()