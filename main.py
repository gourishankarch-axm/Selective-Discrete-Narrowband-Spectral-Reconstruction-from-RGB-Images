

import os
import sys
import argparse
import torch
import numpy as np
import random

from config.config import config, TARGET_BANDS, RGB_BANDS, NUM_INPUT_BANDS, NUM_TARGET_BANDS
from train import train_sr_model
from evaluate import evaluate_water_bodies
from models.unet_model import SRHybridUNet


def main():
    parser = argparse.ArgumentParser(description='RGB to Hyperspectral Reconstruction')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='Mode: train or evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SR HYBRID UNET - RGB to Hyperspectral Reconstruction")
    print("="*70)
    print(f"Input: RGB (bands {RGB_BANDS}) → Output: {NUM_TARGET_BANDS} bands {TARGET_BANDS}")
    print(f"Model Components:")
    print(f"  • 3 Encoder / 3 Decoder stages")
    print(f"  • Advanced Residual Blocks")
    print(f"  • Channel & Spatial Attention")
    print(f"  • Multi-scale Processing")
    print(f"  • Lightweight Transformer (64 tokens at 8×8)")
    print(f"  • Infused Spectral Loss")
    print(f"Training: {config.num_epochs} epochs, batch size {config.batch_size}")
    print(f"Spectral loss weight: {config.spectral_loss_weight}")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check device
    if config.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using CPU")
    
    if args.mode == 'train':
        print("\n[1/1] Training SR Hybrid UNet Model...")
        model = train_sr_model()
        
        if model is None:
            print("Training failed!")
            sys.exit(1)
        
        # Evaluate after training
        print("\nEvaluating trained model on water bodies...")
        evaluate_water_bodies(model)
    
    elif args.mode == 'eval':
        print("\n[1/1] Evaluating SR Hybrid UNet Model...")
        
        # Determine checkpoint path
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = os.path.join(config.model_dir, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at: {checkpoint_path}")
            print("Please train the model first or specify a valid checkpoint path.")
            sys.exit(1)
        
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        model = SRHybridUNet(
            input_channels=NUM_INPUT_BANDS,
            output_channels=NUM_TARGET_BANDS,
            base_features=config.base_features
        ).to(config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'epoch' in checkpoint:
            print(f"Loaded model epoch: {checkpoint['epoch']+1}, loss: {checkpoint['loss']:.6f}")
        
        # Evaluate
        evaluate_water_bodies(model)
    
    print("\n" + "="*70)
    print("SR HYBRID UNET TRAINING AND EVALUATION COMPLETED!")
    print("="*70)
    print(f"Results saved in: {config.results_dir}")
    print(f"Models saved in: {config.model_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
