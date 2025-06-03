import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from new_datasets import data_preprocessing
from model2 import SplitNeRF
from ml_helpers import training
import rendering
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear GPU cache
torch.cuda.empty_cache()

# Dataset and training parameters
data_set_path = '/home/eiyike/DATA/FINAL_DATA'
mode = 'train'
target_size = (400, 400)
batch_size = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
tn = 2
tf = 8
nb_epochs = 30
lr = 1e-3
gamma = 0.5
nb_bins = 100

# Checkpoint settings
checkpoint_dir = 'checkpoints'
checkpoint_interval = 2  # Save checkpoint every 2 epochs
resume_from_checkpoint = True  # Set to True if you want to resume training

# Create checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, checkpoint_path):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss_history': loss_history,
        'hyperparameters': {
            'tn': tn,
            'tf': tf,
            'nb_epochs': nb_epochs,
            'lr': lr,
            'gamma': gamma,
            'nb_bins': nb_bins,
            'batch_size': batch_size,
            'target_size': target_size
        }
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint"""
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint['loss_history']
        
        logger.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch, loss_history
    else:
        logger.info("No checkpoint found, starting training from scratch")
        return 0, []

def get_latest_checkpoint(checkpoint_dir):
    """Get the path to the latest checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
        
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not checkpoint_files:
        return None
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def training_with_checkpoints_wrapper(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, 
                                    dataloader, device, start_epoch=0, existing_loss=[]):
    """
    Wrapper around the original training function to add checkpoint support
    This modifies the training to save checkpoints at specified intervals
    """
    import copy
    
    # Store original training function behavior
    loss_history = existing_loss.copy()
    
    # If starting from scratch or resuming from beginning
    if start_epoch == 0:
        # Use original training function for full training
        logger.info("Using original training function...")
        training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, 
                               nb_epochs, dataloader, device=device)
        
        # Save checkpoint after completion
        final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{nb_epochs}.pth')
        save_checkpoint(model, optimizer, scheduler, nb_epochs-1, training_loss, final_checkpoint_path)
        
        return training_loss
    
    else:
        # Resume training - need to train remaining epochs
        remaining_epochs = nb_epochs - start_epoch
        if remaining_epochs <= 0:
            logger.info("Training already completed!")
            return existing_loss
        
        logger.info(f"Resuming training for {remaining_epochs} more epochs...")
        
        # Create a temporary scheduler for remaining epochs
        # Adjust milestones based on remaining epochs
        adjusted_milestones = [m - start_epoch for m in [5, 10] if m > start_epoch]
        if adjusted_milestones:
            temp_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=adjusted_milestones, gamma=gamma)
        else:
            temp_scheduler = scheduler
        
        # Train for remaining epochs
        remaining_loss = training(model, optimizer, temp_scheduler, tn, tf, nb_bins, 
                                remaining_epochs, dataloader, device=device)
        
        # Combine loss histories
        complete_loss = existing_loss + remaining_loss
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{nb_epochs}.pth')
        save_checkpoint(model, optimizer, scheduler, nb_epochs-1, complete_loss, final_checkpoint_path)
        
        return complete_loss

def periodic_checkpoint_training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, 
                               dataloader, device, start_epoch=0, existing_loss=[]):
    """
    Alternative: Train with periodic checkpoints by splitting epochs
    This breaks training into chunks and saves checkpoints between them
    """
    loss_history = existing_loss.copy()
    current_epoch = start_epoch
    
    while current_epoch < nb_epochs:
        # Calculate how many epochs to train in this chunk
        epochs_to_train = min(checkpoint_interval, nb_epochs - current_epoch)
        
        logger.info(f"Training epochs {current_epoch + 1} to {current_epoch + epochs_to_train}")
        
        # Train for this chunk
        chunk_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, 
                            epochs_to_train, dataloader, device=device)
        
        # Add to loss history
        loss_history.extend(chunk_loss)
        current_epoch += epochs_to_train
        
        # Save checkpoint
        if current_epoch < nb_epochs or current_epoch == nb_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{current_epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, current_epoch-1, loss_history, checkpoint_path)
            
        # Update scheduler state (approximate)
        for _ in range(epochs_to_train):
            scheduler.step()
    
    return loss_history

# Load dataset
logger.info("Loading dataset...")
dataset = data_preprocessing(data_set_path, mode, target_size=target_size)
ray_origins, ray_directions, target_px_values, total_data = dataset.get_rays()

# Prepare data loaders
logger.info("Preparing data loaders...")
size_h, size_w = target_size

# Warm-up data loader (subset of data)
dataloader_warmup = DataLoader(
    torch.cat(
        (torch.from_numpy(ray_origins).reshape(total_data, size_h, size_w, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
         torch.from_numpy(ray_directions).reshape(total_data, size_h, size_w, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
         torch.from_numpy(target_px_values).reshape(total_data, size_h, size_w, 3)[:, 100:300, 100:300, :].reshape(-1, 3)),
        dim=1),
    batch_size=batch_size, shuffle=True
)

# Full data loader
dataloader = DataLoader(
    torch.cat(
        (torch.from_numpy(ray_origins).reshape(-1, 3).type(torch.float),
         torch.from_numpy(ray_directions).reshape(-1, 3).type(torch.float),
         torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)),
        dim=1),
    batch_size=batch_size, shuffle=True
)

# Initialize model, optimizer, and scheduler
logger.info("Initializing model...")
model = SplitNeRF(hidden_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=gamma)

# Load checkpoint if resuming
start_epoch = 0
training_loss = []

if resume_from_checkpoint:
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        start_epoch, training_loss = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)

# Choose training method
USE_PERIODIC_CHECKPOINTS = True  # Set to False to use wrapper method

# Train the model with checkpoint support
logger.info("Starting training...")

if USE_PERIODIC_CHECKPOINTS:
    # Method 1: Break training into chunks with checkpoints between
    training_loss = periodic_checkpoint_training(
        model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader,
        device=device, start_epoch=start_epoch, existing_loss=training_loss
    )
else:
    # Method 2: Use wrapper around original training function
    training_loss = training_with_checkpoints_wrapper(
        model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader,
        device=device, start_epoch=start_epoch, existing_loss=training_loss
    )

# Save the final model
final_model_path = 'nerf_final_model4.pth'
torch.save(model.cpu(), final_model_path)
logger.info(f"Final model saved to '{final_model_path}'.")

# Also save a comprehensive final checkpoint
final_checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
model.to(device)  # Move back to device for checkpoint saving
save_checkpoint(model, optimizer, scheduler, nb_epochs-1, training_loss, final_checkpoint_path)

# Save training metadata
metadata = {
    'total_epochs': nb_epochs,
    'final_loss': training_loss[-1] if training_loss else None,
    'total_loss_points': len(training_loss),
    'hyperparameters': {
        'tn': tn,
        'tf': tf,
        'lr': lr,
        'gamma': gamma,
        'nb_bins': nb_bins,
        'batch_size': batch_size,
        'target_size': target_size
    },
    'model_path': final_model_path,
    'checkpoint_path': final_checkpoint_path,
    'training_method': 'periodic_checkpoints' if USE_PERIODIC_CHECKPOINTS else 'wrapper_method'
}

with open(os.path.join(checkpoint_dir, 'training_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# Plot and save training loss
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(training_loss)
plt.xlabel('Training Step/Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)

# If we have many loss points, also show moving average
if len(training_loss) > 50:
    window_size = max(10, len(training_loss) // 20)
    moving_avg = np.convolve(training_loss, np.ones(window_size)/window_size, mode='valid')
    plt.subplot(2, 1, 2)
    plt.plot(moving_avg)
    plt.xlabel('Training Step/Epoch')
    plt.ylabel('Loss (Moving Average)')
    plt.title(f'Training Loss Moving Average (window={window_size})')
    plt.grid(True)

plt.tight_layout()
plt.savefig("training_loss.png", dpi=150, bbox_inches='tight')
plt.show()
logger.info("Training loss plot saved to 'training_loss.png'.")

# Print training summary
logger.info("\n" + "="*60)
logger.info("TRAINING SUMMARY")
logger.info("="*60)
logger.info(f"Total epochs: {nb_epochs}")
logger.info(f"Epochs completed: {start_epoch} -> {nb_epochs}")
logger.info(f"Total loss data points: {len(training_loss)}")
logger.info(f"Final loss: {training_loss[-1]:.6f}" if training_loss else "No training completed")
logger.info(f"Model saved to: {final_model_path}")
logger.info(f"Checkpoints saved in: {checkpoint_dir}")
logger.info(f"Training method: {'Periodic Checkpoints' if USE_PERIODIC_CHECKPOINTS else 'Wrapper Method'}")
logger.info("="*60)

# List saved checkpoints
if os.path.exists(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoints:
        logger.info(f"Available checkpoints: {', '.join(sorted(checkpoints))}")

# Optional: Render an image using the trained model
if False:  # Set to True to enable rendering
    logger.info("Rendering test image...")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        img = rendering.rendering(
            model,
            torch.from_numpy(ray_origins[0]).type(torch.float32).to(device),
            torch.from_numpy(ray_directions[0]).type(torch.float32).to(device),
            tn, tf, nb_bins=100, device=device
        )
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img.reshape(size_h, size_w, 3).data.cpu().numpy())
    plt.axis('off')
    plt.title('Rendered Image')
    plt.savefig("rendered_image.png", bbox_inches='tight', dpi=150)
    plt.show()
    logger.info("Rendered image saved to 'rendered_image.png'.")

logger.info("Training script completed successfully!")