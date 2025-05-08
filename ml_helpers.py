from tqdm import tqdm
from rendering import rendering
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device='cpu'):
    """
    Train a NeRF model.

    Args:
        model (torch.nn.Module): The NeRF model to train.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        tn (float): Near bound for ray sampling.
        tf (float): Far bound for ra       breakpoint()y sampling.
        nb_bins (int): Number of bins for hierarchical sampling.
        nb_epochs (int): Number of training epochs.
        data_loader (torch.utils.data.DataLoader): Data loader providing batches of rays and targets.
        device (str): Device for computation ('cpu' or 'cuda').

    Returns:
        list: A list of training loss values.
    """
    training_loss = []
    model.to(device)

    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{nb_epochs}")

        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            ray_origins = batch[:, :3].to(device)  # Ray origins
            
            ray_directions = batch[:, 3:6].to(device)  # Ray directions
            target = batch[:, 6:].to(device)  # Target pixel values

            # Forward pass
            prediction = rendering(model, ray_origins, ray_directions, tn, tf, nb_bins=nb_bins, device=device)

            # Compute loss
            loss = ((prediction - target) ** 2).mean()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss
            epoch_loss += loss.item()
            training_loss.append(loss.item())

            # Log progress
            if len(training_loss) % 10 == 0:
                logger.info(f"Batch {len(training_loss)}: Loss = {loss.item():.6f}")

        # Update learning rate
        scheduler.step()

        # Log epoch loss
        logger.info(f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss / len(data_loader):.6f}")

        # Save model checkpoint
        torch.save(model.cpu(), f'nerf_model_epoch_newwww{epoch + 1}.pth')
        # torch.save(model.state_dict(), f'nerf_model_epoch_{epoch + 1}.pth')
        model.to(device)
        logger.info(f"Model checkpoint saved for epoch {epoch + 1}.")

    return training_loss