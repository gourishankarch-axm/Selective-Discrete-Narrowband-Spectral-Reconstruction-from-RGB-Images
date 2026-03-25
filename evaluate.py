
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import config, TARGET_BANDS, RGB_BANDS, NUM_INPUT_BANDS, NUM_TARGET_BANDS
from data.dataset import SpectralDataset
from models.unet_model import SRHybridUNet
from utils.metrics import calculate_metrics, save_metrics
from utils.visualization import create_visualization


def evaluate_water_bodies(model):
   
    print(f"\n" + "="*60)
    print(f"Evaluating SR Hybrid UNet on Water Bodies")
    print("="*60)
    
    if not os.path.exists(config.water_bodies_dir):
        print(f"Water bodies directory not found: {config.water_bodies_dir}")
        return None, None
    
    # Load water bodies data
    dataset = SpectralDataset(
        config.water_bodies_dir,
        target_bands=TARGET_BANDS,
        rgb_bands=RGB_BANDS,
        augment=False,
        max_samples=50
    )
    
    if len(dataset) == 0:
        print("No water bodies data found!")
        return None, None
    
    print(f"Loaded {len(dataset)} water bodies samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=False,
        num_workers=0
    )
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for lr_batch, hr_batch in tqdm(dataloader, desc="Evaluating"):
            lr_batch = lr_batch.to(config.device)
            hr_batch = hr_batch.to(config.device)
            
            predictions = model(lr_batch)
            all_predictions.append(predictions.cpu())
            all_targets.append(hr_batch.cpu())
    
    if not all_predictions:
        print("No predictions made!")
        return None, None
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    
    print(f"\nEvaluation Metrics:")
    print(f"MSE:   {metrics['mse']:.6f}")
    print(f"MAE:   {metrics['mae']:.6f}")
    print(f"RMSE:  {metrics['rmse']:.6f}")
    print(f"PSNR:  {metrics['psnr']:.2f} dB")
    
    # Save metrics
    metrics_file = os.path.join(config.results_dir, 'water_bodies_metrics.txt')
    save_metrics(metrics, metrics_file, len(predictions), TARGET_BANDS)
    
    # Create visualizations
    create_visualization(predictions, targets, TARGET_BANDS, config.results_dir, metrics)
    
    return predictions, targets


if __name__ == "__main__":
    # Load model
    best_model_path = os.path.join(config.model_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        print(f"Loading model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=config.device)
        
        model = SRHybridUNet(
            input_channels=NUM_INPUT_BANDS,
            output_channels=NUM_TARGET_BANDS,
            base_features=config.base_features
        ).to(config.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model epoch: {checkpoint['epoch']+1}, loss: {checkpoint['loss']:.6f}")
        
        # Evaluate
        evaluate_water_bodies(model)
    else:
        print(f"Model not found at: {best_model_path}")
