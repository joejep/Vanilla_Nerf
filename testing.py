import torch
import numpy as np
from PIL import Image
import os
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from new_datasets import data_preprocessing
from model2 import SplitNeRF
import rendering
# import rendering2
import logging
import cProfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear GPU cache
torch.cuda.empty_cache()

# Dataset and training parameters
data_set_path = '/home/eiyike/DATA/FINAL_DATA'
mode = 'test'
target_size = (400, 400)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tn = 2
tf = 6

# Load dataset
logger.info("Loading test dataset...")
dataset = data_preprocessing(data_set_path, mode, target_size=target_size)
test_o, test_d, target_px_values, total_data = dataset.get_rays()

def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint file
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model with same architecture
    model = SplitNeRF(hidden_dim=128).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Log checkpoint info
    if 'epoch' in checkpoint:
        logger.info(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    if 'loss_history' in checkpoint and checkpoint['loss_history']:
        logger.info(f"Final training loss: {checkpoint['loss_history'][-1]:.6f}")
    
    return model

def load_model_from_direct_save(model_path, device):
    """
    Load model from direct model save (not checkpoint)
    """
    logger.info(f"Loading model from direct save: {model_path}")
    return torch.load(model_path, map_location=device, weights_only=False).to(device)

# Try loading from checkpoint first, then fallback to direct model file
checkpoint_path = '/home/eiyike/Documents/Vanilla_Nerf/checkpoints/final_checkpoint.pth'
direct_model_path = '/home/eiyike/Documents/Vanilla_Nerf/nerf_final_model4.pth'

try:
    # First try loading from checkpoint
    if os.path.exists(checkpoint_path):
        model = load_model_from_checkpoint(checkpoint_path, device)
    elif os.path.exists(direct_model_path):
        model = load_model_from_direct_save(direct_model_path, device)
    else:
        raise FileNotFoundError("No model file found. Please check the paths.")
        
except Exception as e:
    logger.error(f"Error loading from checkpoint: {e}")
    logger.info("Trying to load from direct model save...")
    
    if os.path.exists(direct_model_path):
        model = load_model_from_direct_save(direct_model_path, device)
    else:
        raise FileNotFoundError("No model file found. Please check the paths.")

model.eval()  # Set model to evaluation mode
logger.info("Model loaded successfully!")

# Function to convert MSE to PSNR
def mse2psnr(mse):
    return 20 * np.log10(1 / np.sqrt(mse))

@torch.no_grad()
def test(model, ray_origins, ray_directions, tn, tf, nb_bins=100, chunk_size=10, H=400, W=400, target=None):
    """
    Render images using the NeRF model and compute evaluation metrics.

    Args:
        model (torch.nn.Module): Trained NeRF model.
        ray_origins (torch.Tensor): Ray origins of shape (num_rays, 3).
        ray_directions (torch.Tensor): Ray directions of shape (num_rays, 3).
        tn (float): Near bound for ray sampling.
        tf (float): Far bound for ray sampling.
        nb_bins (int): Number of bins for hierarchical sampling.
        chunk_size (int): Number of rays to process at once.
        H (int): Height of the rendered image.
        W (int): Width of the rendered image.
        target (np.ndarray): Ground truth image for evaluation.

    Returns:
        image (np.ndarray): Rendered image of shape (H, W, 3).
        mse (float): Mean squared error between the rendered and target images.
        psnr (float): Peak signal-to-noise ratio between the rendered and target images.
    """
    ray_origins = ray_origins.chunk(chunk_size)
    ray_directions = ray_directions.chunk(chunk_size)
    
    # Handle rendering2 import gracefully
    try:
        import rendering2
        use_rendering2 = True
    except ImportError:
        logger.warning("rendering2 module not found. Skipping 3D point extraction.")
        use_rendering2 = False
    
    xyzs = []
    image = []
    
    for o_batch, d_batch in zip(ray_origins, ray_directions):
        img_batch = rendering.rendering(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device=o_batch.device)
        
        if use_rendering2:
            xyz = rendering2.rendering(model, o_batch, d_batch, tn, tf, nb_bins=nb_bins, device=o_batch.device)
            xyzs.append(xyz)
            
        image.append(img_batch)  # [chunk_size, 3]
    
    image = torch.cat(image)
    image = image.reshape(H, W, 3).cpu().numpy()
    
    if use_rendering2:
        xyzs = torch.cat(xyzs)
        xyzs = xyzs.reshape(H, W, 100, 3).cpu().numpy()
    else:
        xyzs = None

    if target is not None:
        mse = ((image - target) ** 2).mean()
        psnr = mse2psnr(mse)
        return image, xyzs, mse, psnr
    else:
        return image, xyzs

# Profile the test function
logger.info("Starting model evaluation...")
with cProfile.Profile() as pr:
    result = test(
        model,
        torch.from_numpy(test_o[4]).to(device).float(),
        torch.from_numpy(test_d[4]).to(device).float(),
        tn, tf, nb_bins=100, chunk_size=10, target=target_px_values[4].reshape(400, 400, 3)
    )
    pr.dump_stats('prof_test_objectscene10_epochs.prof')

# Unpack results
if len(result) == 4:
    img, xyzs, mse, psnr = result
else:
    img, xyzs = result
    mse, psnr = None, None

# Log evaluation metrics
if mse is not None and psnr is not None:
    logger.info(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
else:
    logger.info("No target image provided for evaluation")

# Visualize 3D points if available
if xyzs is not None:
    logger.info("Visualizing 3D point data...")
    
    z_index = 50  # You can change this to any value between 0 and 99
    # Choose which component to visualize (0, 1, or 2 from the last dimension)
    component = 2  # Change to 1 or 2 to see other components

    # Extract a 2D slice
    slice_2d = xyzs[:, :, z_index, component]

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(slice_2d, cmap='viridis')
    plt.colorbar(label=f'Component {component} value')
    plt.title(f'2D Slice at z_index={z_index}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(f'slice_2d_component_{component}.png')
    plt.show()

    # Alternatively, you could visualize an average across the 3rd dimension
    avg_2d = np.mean(xyzs[:, :, :, component], axis=2)

    plt.figure(figsize=(10, 8))
    plt.imshow(avg_2d, cmap='plasma')
    plt.colorbar(label=f'Average of component {component}')
    plt.title(f'Average of Component {component} Across Z Dimension')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(f'avg_2d_component_{component}.png')
    plt.show()

    # 3D scatter plot (subsampled for performance)
    logger.info("Creating 3D scatter plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # The full dataset has 400*400*100 = 16,000,000 points, which is too many to plot
    # Let's subsample to make the visualization manageable
    # Taking every 20th point in each dimension
    step = 20
    x = xyzs[::step, ::step, ::step, 0].flatten()
    y = xyzs[::step, ::step, ::step, 1].flatten()
    z = xyzs[::step, ::step, ::step, 2].flatten()

    # Create the 3D scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.8)

    # Add labels and colorbar
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.colorbar(scatter, ax=ax, label='Z value')
    plt.title('3D Scatter Plot of Points (Subsampled)')

    # Show the plot
    plt.tight_layout()
    plt.savefig('3d_scatter_plot.png')
    plt.show()

# Save and display the rendered image
plt.figure(figsize=(10, 8))
plt.imshow(img)
if psnr is not None:
    plt.title(f"Image Generated by NeRF (PSNR: {psnr:.2f} dB)")
else:
    plt.title("Image Generated by NeRF")
plt.axis('off')
plt.savefig("dragon5555.png", bbox_inches='tight', dpi=150)
plt.show()

logger.info("Evaluation completed successfully!")
logger.info(f"Rendered image saved as 'dragon5555.png'")
if xyzs is not None:
    logger.info("3D visualization plots saved")

# Optional: Additional rendering test
if False:  # Set to True to enable additional rendering
    logger.info("Rendering additional test image...")
    size_h, size_w = target_size
    img = rendering.rendering(
        model,
        torch.from_numpy(test_o[0]).type(torch.float32).to(device),
        torch.from_numpy(test_d[0]).type(torch.float32).to(device),
        tn, tf, nb_bins=100, device=device
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(img.reshape(size_h, size_w, 3).data.cpu().numpy())
    plt.title("Additional Rendered Image")
    plt.axis('off')
    plt.savefig("additional_rendered_image.png", bbox_inches='tight')
    plt.show()
    logger.info("Additional rendered image saved to 'additional_rendered_image.png'.")