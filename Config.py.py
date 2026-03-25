"""
Configuration file for RGB to Hyperspectral Reconstruction
"""

import torch

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Target bands (0-indexed) - 6 bands to predict
TARGET_BANDS = [6, 7, 10, 12, 16, 21]
NUM_TARGET_BANDS = len(TARGET_BANDS)

# RGB bands - select 3 bands to serve as RGB input
RGB_BANDS = [20, 11, 5]
NUM_INPUT_BANDS = len(RGB_BANDS)


class Config:
    """Main configuration class"""
    
    # ========== DATASET PATHS ==========
    train_data = '/path/to/train_images'      # Update this path
    test_data = '/path/to/test_images'        # Update this path
    water_bodies_dir = '/path/to/water_bodies' # Update this path
    
    # ========== TRAINING PARAMETERS ==========
    num_epochs = 100
    batch_size = 16
    lr = 0.0004
    val_split_ratio = 0.2
    spectral_loss_weight = 0.3
    
    # ========== MODEL SAVE PATHS ==========
    model_dir = 'NB_SR_Model'
    results_dir = 'NB_SR_results'
    
    # ========== DEVICE ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========== VISUALIZATION ==========
    num_visual_samples = 3
    
    # ========== MODEL ARCHITECTURE ==========
    base_features = 16          # Initial features after first conv
    transformer_dim = 96        # Transformer hidden dimension
    num_tokens = 64             # 8×8 = 64 tokens


config = Config()