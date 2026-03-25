
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
 
    
    def __init__(self, data_dir, target_bands, rgb_bands, augment=False, max_samples=None):
       
        self.data_dir = data_dir
        self.target_bands = target_bands
        self.rgb_bands = rgb_bands
        self.augment = augment
        self.max_samples = max_samples
        
        # Find all .npy files
        print(f"Scanning {data_dir}...")
        self.file_paths = []
        for ext in ['*.npy', '*.NPY']:
            self.file_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        
        print(f"Found {len(self.file_paths)} files")
        
        # Limit samples if specified
        if max_samples and len(self.file_paths) > max_samples:
            print(f"Limiting to {max_samples} samples for faster processing")
            self.file_paths = self.file_paths[:max_samples]
        
        # Cache for loaded data
        self.data_cache = {}
        self.loaded_indices = []
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        if idx in self.data_cache:
            return self.data_cache[idx]
        
        file_path = self.file_paths[idx]
        
        try:
            # Load data
            data = np.load(file_path, allow_pickle=True).astype(np.float32)
            
            # Ensure correct shape
            if data.shape != (64, 64, 153):
                if data.size == 64 * 64 * 153:
                    data = data.reshape(64, 64, 153)
                else:
                    raise ValueError(f"Incorrect size: {data.shape}")
            
            # Extract RGB input and target bands
            rgb = data[:, :, self.rgb_bands]      # Shape: (64, 64, 3)
            hr = data[:, :, self.target_bands]    # Shape: (64, 64, 6)
            
            # Convert to channel-first format
            rgb = rgb.transpose(2, 0, 1)          # Shape: (3, 64, 64)
            hr = hr.transpose(2, 0, 1)            # Shape: (6, 64, 64)
            
            # Convert to tensors
            rgb_tensor = torch.from_numpy(rgb.copy())
            hr_tensor = torch.from_numpy(hr.copy())
            
            # Normalize each channel to [0, 1]
            def normalize_channel(channel):
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    return (channel - min_val) / (max_val - min_val)
                return channel * 0
            
            for c in range(rgb_tensor.shape[0]):
                rgb_tensor[c] = normalize_channel(rgb_tensor[c])
            for c in range(hr_tensor.shape[0]):
                hr_tensor[c] = normalize_channel(hr_tensor[c])
            
            # Add small noise for numerical stability
            rgb_tensor = rgb_tensor + torch.randn_like(rgb_tensor) * 1e-6
            hr_tensor = hr_tensor + torch.randn_like(hr_tensor) * 1e-6
            
            # Apply augmentation if enabled
            if self.augment:
                rgb_tensor, hr_tensor = self._apply_augmentation(rgb_tensor, hr_tensor)
            
            # Cache the result
            self.data_cache[idx] = (rgb_tensor, hr_tensor)
            self.loaded_indices.append(idx)
            
            # Limit cache size
            if len(self.data_cache) > 100:
                oldest_idx = self.loaded_indices.pop(0)
                del self.data_cache[oldest_idx]
            
            return rgb_tensor, hr_tensor
        
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            # Return dummy data on error
            num_rgb = len(self.rgb_bands)
            num_target = len(self.target_bands)
            return torch.zeros((num_rgb, 64, 64)), torch.zeros((num_target, 64, 64))
    
    def _apply_augmentation(self, rgb, hr):
        """Apply random horizontal/vertical flips"""
        if random.random() > 0.5:
            rgb = torch.flip(rgb, dims=[1])
            hr = torch.flip(hr, dims=[1])
        if random.random() > 0.5:
            rgb = torch.flip(rgb, dims=[2])
            hr = torch.flip(hr, dims=[2])
        return rgb, hr
