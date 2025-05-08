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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear GPU cache
torch.cuda.empty_cache()

# Dataset and training parameters
data_set_path = '/home/eiyike/DATA/vanilla_dataset'
mode = 'train'
target_size = (400, 400)
batch_size = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
tn = 2
tf = 8
nb_epochs = 5
lr = 1e-3
gamma = 0.5
nb_bins = 100

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

# Train the model
logger.info("Starting training...")
training_loss = training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, dataloader, device=device)

# Save the final model
torch.save(model.cpu(), 'nerf_final_model2.pth')
logger.info("Model saved to 'nerf_final_model2.pth'.")

# Plot and save training loss
plt.plot(training_loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig("training_loss.png")
plt.show()
logger.info("Training loss plot saved to 'training_loss.png'.")

# Optional: Render an image using the trained model
if False:  # Set to True to enable rendering
    logger.info("Rendering test image...")
    img = rendering.rendering(
        model,
        torch.from_numpy(ray_origins[0]).type(torch.float32).to(device),
        torch.from_numpy(ray_directions[0]).type(torch.float32).to(device),
        tn, tf, nb_bins=100, device=device
    )
    plt.imshow(img.reshape(size_h, size_w, 3).data.cpu().numpy())
    plt.savefig("rendered_image.png")
    plt.show()
    logger.info("Rendered image saved to 'rendered_image.png'.")