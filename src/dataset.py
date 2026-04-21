import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class CoralMultimodalDataset(Dataset):
    def __init__(self, df, transform=None, env_stats=None):
        """
        Initialize the Multimodal Dataset (Image + Environment).
        
        Args:
            df (pd.DataFrame): DataFrame containing the split data (Train or Val/Test).
            transform (callable, optional): Image transformations (Resize, Normalize, Augmentation).
            env_stats (dict): Dictionary containing mean and std values calculated ONLY FROM the Train set.
        """
        # Ensure index is reset so __getitem__ accessing .iloc does not misalign rows
        self.data = df.reset_index(drop=True)
        self.transform = transform
        
        # env_stats is mandatory for safe data normalization
        if env_stats is None:
            raise ValueError("ERROR: env_stats (calculated from the Train set) must be provided to avoid Data Leakage!")
        
        self.env_stats = env_stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # ==========================================
        # 1. PROCESS VISUAL BRANCH (Visual Data)
        # ==========================================
        img_path = row['Image_Path']
        
        # Convert to RGB in case the original image is Grayscale or has an Alpha channel (RGBA)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # ==========================================
        # 2. PROCESS ENVIRONMENTAL BRANCH (Environmental Data)
        # ==========================================
        # Use parameters from the Train set (env_stats) for Z-score normalization
        sst_norm = (row['SST (°C)'] - self.env_stats['sst_mean']) / (self.env_stats['sst_std'] + 1e-8)
        ph_norm = (row['pH Level'] - self.env_stats['ph_mean']) / (self.env_stats['ph_std'] + 1e-8)
        lat_norm = (row['Latitude'] - self.env_stats['lat_mean']) / (self.env_stats['lat_std'] + 1e-8)
        lon_norm = (row['Longitude'] - self.env_stats['lon_mean']) / (self.env_stats['lon_std'] + 1e-8)
        month_norm = (row['Month'] - self.env_stats['month_mean']) / (self.env_stats['month_std'] + 1e-8)
        
        # Marine Heatwave feature is binary (0 or 1), so keep it as is
        heatwave = float(row['Marine Heatwave']) 
        
        # Combine into a 6-dimensional vector
        env_features = torch.tensor(
            [sst_norm, ph_norm, lat_norm, lon_norm, month_norm, heatwave], 
            dtype=torch.float32
        )
        
        # ==========================================
        # 3. EXTRACT LABEL (Label)
        # ==========================================
        # 0: Healthy, 1: Bleached
        label = torch.tensor(row['Label'], dtype=torch.long)
        
        return image, env_features, label