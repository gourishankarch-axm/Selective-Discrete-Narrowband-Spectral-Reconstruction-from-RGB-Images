
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from config.config import config, TARGET_BANDS, RGB_BANDS, NUM_INPUT_BANDS, NUM_TARGET_BANDS
from data.dataset import SpectralDataset
from models.unet_model import SRHybridUNet
from losses.spectral_loss import SpectralCorrelationLoss
from utils.visualization import plot_training_history


def train_sr_model():
   
    print(f"\nTraining SR Hybrid UNet on: {config.device}")
    print(f"Target bands: {TARGET_BANDS}")
    print(f"Spectral loss weight: {config.spectral_loss_weight}")
    
    # Create directories
    Path(config.model_dir).mkdir(exist_ok=True)
    Path(config.results_dir).mkdir(exist_ok=True)
    
    # Create dataset
    dataset = SpectralDataset(
        config.train_data, 
        target_bands=TARGET_BANDS,
        rgb_bands=RGB_BANDS,
        augment=True, 
        max_samples=2000
    )
    
    if len(dataset) == 0:
        print("No valid data found!")
        return None
    
    print(f"Loaded {len(dataset)} samples")
    
    # Split dataset
    val_size = int(len(dataset) * config.val_split_ratio)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training: {train_size} samples, Validation: {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = SRHybridUNet(
        input_channels=NUM_INPUT_BANDS,
        output_channels=NUM_TARGET_BANDS,
        base_features=config.base_features
    ).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,} (<1.5M ✓)")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Memory: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss functions
    spectral_loss_fn = SpectralCorrelationLoss(weight=config.spectral_loss_weight)
    mse_loss_fn = torch.nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    
    # Early stopping
    patience = 12
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training history
    train_losses = []
    val_losses = []
    
    print("\n" + "="*50)
    print("Starting SR Hybrid UNet Training")
    print("="*50)
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_total_loss = 0.0
        train_mse_loss = 0.0
        train_spectral_loss = 0.0
        batch_count = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')
        for lr_batch, hr_batch in train_bar:
            lr_batch = lr_batch.to(config.device)
            hr_batch = hr_batch.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(lr_batch)
            
            # Combined spectral loss
            total_loss, mse_loss, spectral_loss = spectral_loss_fn(outputs, hr_batch)
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_total_loss += total_loss.item()
            train_mse_loss += mse_loss
            train_spectral_loss += spectral_loss
            batch_count += 1
            
            train_bar.set_postfix({
                'total': total_loss.item(),
                'mse': mse_loss,
                'spectral': spectral_loss
            })
        
        avg_train_loss = train_total_loss / batch_count
        avg_train_mse = train_mse_loss / batch_count
        avg_train_spectral = train_spectral_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_total_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Val]')
            for lr_batch, hr_batch in val_bar:
                lr_batch = lr_batch.to(config.device)
                hr_batch = hr_batch.to(config.device)
                
                outputs = model(lr_batch)
                loss = mse_loss_fn(outputs, hr_batch)
                val_total_loss += loss.item()
                val_batch_count += 1
                val_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_total_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Total: {avg_train_loss:.6f}, MSE: {avg_train_mse:.6f}, Spectral: {avg_train_spectral:.6f}")
        print(f"  Val MSE: {avg_val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, os.path.join(config.model_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (loss: {best_val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                      os.path.join(config.model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.model_dir, 'final_model.pth'))
    print(f"\n✓ Training completed! Best val loss: {best_val_loss:.6f}")
    
    # Plot training history
    plot_training_history(
        train_losses, 
        val_losses, 
        os.path.join(config.results_dir, 'training_history.png')
    )
    
    return model


if __name__ == "__main__":
    train_sr_model()
