import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def build_multimodal_dataset(img_dir_healthy, img_dir_bleached, csv_path, output_path):
    print("Initializing Multimodal data (Data Leakage prevention mode)...")

    # ==========================================
    # 1. READ AND PREPROCESS ENVIRONMENTAL DATA
    # ==========================================
    env_df = pd.read_csv(csv_path)

    # Clean label column
    env_df['Bleaching Severity'] = env_df['Bleaching Severity'].fillna('None').astype(str).str.strip().str.title()
    
    # Extract basic features
    env_df['Month'] = pd.to_datetime(env_df['Date']).dt.month
    env_df['Marine Heatwave'] = env_df['Marine Heatwave'].astype(int)
    
    # Separate into 2 environmental groups
    healthy_env = env_df[env_df['Bleaching Severity'] == 'None'].copy()
    bleached_env = env_df[env_df['Bleaching Severity'] != 'None'].copy()

    if len(healthy_env) == 0 or len(bleached_env) == 0:
        print("Error: Data filtering failed due to missing corresponding labels!")
        return

    # ==========================================
    # 2. SCAN IMAGE LISTS
    # ==========================================
    healthy_imgs = [f for f in os.listdir(img_dir_healthy) if f.endswith(('.jpg', '.png', '.jpeg'))]
    bleached_imgs = [f for f in os.listdir(img_dir_bleached) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # ==========================================
    # 3. SPLIT TRAIN/VAL BEFORE MAPPING (CRITICAL FIX)
    # ==========================================
    # This ensures 100% no tabular row or image is duplicated between Train and Val
    
    # Split Tabular Data
    h_env_train, h_env_val = train_test_split(healthy_env, test_size=0.2, random_state=42)
    b_env_train, b_env_val = train_test_split(bleached_env, test_size=0.2, random_state=42)

    # Split Visual Data
    h_img_train, h_img_val = train_test_split(healthy_imgs, test_size=0.2, random_state=42)
    b_img_train, b_img_val = train_test_split(bleached_imgs, test_size=0.2, random_state=42)

    # ==========================================
    # 4. DATA MAPPING FUNCTION
    # ==========================================
    def map_data(env_data, img_list, label, split_name, img_dir):
        if len(img_list) == 0 or len(env_data) == 0:
            return pd.DataFrame()
        
        # Randomly sample Tabular data (with replacement) to match the number of images IN EACH SPLIT
        mapped = env_data.sample(n=len(img_list), replace=True, random_state=42).reset_index(drop=True)
        mapped['Image_ID'] = img_list
        mapped['Image_Path'] = [os.path.join(img_dir, img) for img in img_list]
        mapped['Label'] = label
        mapped['Split'] = split_name  # Mark as Train or Val
        return mapped

    # Map Train set
    train_h = map_data(h_env_train, h_img_train, 0, 'Train', img_dir_healthy)
    train_b = map_data(b_env_train, b_img_train, 1, 'Train', img_dir_bleached)
    
    # Map Validation set
    val_h = map_data(h_env_val, h_img_val, 0, 'Val', img_dir_healthy)
    val_b = map_data(b_env_val, b_img_val, 1, 'Val', img_dir_bleached)

    # ==========================================
    # 5. MERGE AND SAVE FILES
    # ==========================================
    final_df = pd.concat([train_h, train_b, val_h, val_b], ignore_index=True)
    
    # Shuffle randomly
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True) 

    cols_to_keep = ['Image_ID', 'Image_Path', 'SST (°C)', 'pH Level', 
                    'Latitude', 'Longitude', 'Month', 'Marine Heatwave', 'Label', 'Split']
    final_df = final_df[cols_to_keep]

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save total file
    final_df.to_csv(output_path, index=False)
    
    # Export Train and Val files for direct use in train.py
    train_out = os.path.join(out_dir, 'train_split.csv')
    val_out = os.path.join(out_dir, 'val_split.csv')
    
    final_df[final_df['Split'] == 'Train'].to_csv(train_out, index=False)
    final_df[final_df['Split'] == 'Val'].to_csv(val_out, index=False)

    print(f"Complete! Data structure is completely secured:")
    print(f"  -> Total: {len(final_df)} samples ({output_path})")
    print(f"  -> Train set: {len(train_h) + len(train_b)} samples -> {train_out}")
    print(f"  -> Validation set: {len(val_h) + len(val_b)} samples -> {val_out}")

if __name__ == "__main__":
    build_multimodal_dataset(
        img_dir_healthy='data/raw/healthy_corals',
        img_dir_bleached='data/raw/bleached_corals',
        csv_path='data/raw/realistic_ocean_climate_dataset.csv',
        output_path='data/processed/multimodal_dataset.csv'
    )