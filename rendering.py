# # import torch
# # import logging

# # # Set up logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # # def compute_accumulated_transmittance(betas):
# # #     accumulated_transmittance = torch.cumprod(betas, 1)     
# # #     return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
# # #                       accumulated_transmittance[:, :-1]), dim=1)

# # def compute_accumulated_transmittance(betas):
# #     """
# #     Compute the accumulated transmittance along the rays.

# #     Args:
# #         betas (torch.Tensor): Transmittance values of shape (num_rays, num_bins).

# #     Returns:
# #         torch.Tensor: Accumulated transmittance of shape (num_rays, num_bins).
# #     """
# #     accumulated_transmittance = torch.cumprod(betas, 1)
# #     return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
# #                      accumulated_transmittance[:, :-1]), dim=1)
# # def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
# #     """
# #     Render the color of rays using volume rendering.

# #     Args:
# #         model (torch.nn.Module): The NeRF model.
# #         rays_o (torch.Tensor): Ray origins of shape (num_rays, 3).
# #         rays_d (torch.Tensor): Ray directions of shape (num_rays, 3).
# #         tn (float): Near bound for ray sampling.
# #         tf (float): Far bound for ray sampling.
# #         nb_bins (int): Number of bins for sampling along the rays.
# #         device (str): Device for computation ('cpu' or 'cuda').
# #         white_bckgr (bool): Whether to assume a white background.

# #     Returns:
# #         torch.Tensor: Rendered colors of shape (num_rays, 3).
# #     """
# #     # Sample points along the rays
# #     t = torch.linspace(tn, tf, nb_bins, device=device)  # [nb_bins]
# #     delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))  # [nb_bins]

# #     # Compute sampled points along the rays
# #     sampled_points = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1)  # [num_rays, nb_bins, 3]

# #     # Predict colors and densities
# #     colors, density = model.intersect(
# #         sampled_points.reshape(-1, 3),
# #         rays_d.expand(nb_bins, rays_o.shape[0], 3).transpose(0, 1).reshape(-1, 3)
# #     )
# #     colors = colors.reshape((rays_o.shape[0], nb_bins, 3))  # [num_rays, nb_bins, 3]
# #     density = density.reshape((rays_o.shape[0], nb_bins))  # [num_rays, nb_bins]

# #     # Compute alpha and weights
# #     alpha = 1 - torch.exp(-density * delta.unsqueeze(0))  # [num_rays, nb_bins]
# #     weights = compute_accumulated_transmittance(1 - alpha) * alpha  # [num_rays, nb_bins]

# #     # Compute final colors
# #     if white_bckgr:
# #         c = (weights.unsqueeze(-1) * colors).sum(1)  # [num_rays, 3]
# #         weight_sum = weights.sum(-1)  # [num_rays]
# #         return c + 1 - weight_sum.unsqueeze(-1)
# #     else:
# #         return (weights.unsqueeze(-1) * colors).sum(1)  # [num_rays, 3]


# import torch
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def compute_accumulated_transmittance(betas):
#     """
#     Compute the accumulated transmittance along the rays.
    
#     Args:
#         betas (torch.Tensor): Transmittance values of shape (num_rays, num_bins).
    
#     Returns:
#         torch.Tensor: Accumulated transmittance of shape (num_rays, num_bins).
#     """
#     accumulated_transmittance = torch.cumprod(betas, 1)
#     return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=accumulated_transmittance.device),
#                      accumulated_transmittance[:, :-1]), dim=1)

# def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
#     """
#     Render the color of rays using volume rendering.
    
#     Args:
#         model (torch.nn.Module): The NeRF model.
#         rays_o (torch.Tensor): Ray origins of shape (num_rays, 3).
#         rays_d (torch.Tensor): Ray directions of shape (num_rays, 3).
#         tn (float): Near bound for ray sampling.
#         tf (float): Far bound for ray sampling.
#         nb_bins (int): Number of bins for sampling along the rays.
#         device (str): Device for computation ('cpu' or 'cuda').
#         white_bckgr (bool): Whether to assume a white background.
    
#     Returns:
#         torch.Tensor: Rendered colors of shape (num_rays, 3).
#     """
#     # Sample points along the rays
#     t = torch.linspace(tn, tf, nb_bins, device=device)  # [nb_bins]
#     delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))  # [nb_bins]
    
#     # Compute sampled points along the rays
#     sampled_points = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1)  # [num_rays, nb_bins, 3]
    
#     # Reshape points and directions for network input
#     flat_points = sampled_points.reshape(-1, 3)
#     expanded_dirs = rays_d.expand(nb_bins, rays_o.shape[0], 3).transpose(0, 1).reshape(-1, 3)
    
#     # Predict densities and colors separately
#     density = model.density(flat_points)
#     colors = model.color(flat_points, expanded_dirs)
    
#     # Reshape outputs
#     colors = colors.reshape((rays_o.shape[0], nb_bins, 3))  # [num_rays, nb_bins, 3]
#     density = density.reshape((rays_o.shape[0], nb_bins))  # [num_rays, nb_bins]
    
#     # Compute alpha and weights
#     alpha = 1 - torch.exp(-density * delta.unsqueeze(0))  # [num_rays, nb_bins]
#     weights = compute_accumulated_transmittance(1 - alpha) * alpha  # [num_rays, nb_bins]
    
#     # Compute final colors
#     if white_bckgr:
#         c = (weights.unsqueeze(-1) * colors).sum(1)  # [num_rays, 3]
#         weight_sum = weights.sum(-1)  # [num_rays]
#         return c + 1 - weight_sum.unsqueeze(-1)
#     else:
#         return (weights.unsqueeze(-1) * colors).sum(1)  # [num_rays, 3]



import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_accumulated_transmittance(betas):
    """
    Compute the accumulated transmittance along the rays.
    
    Args:
        betas (torch.Tensor): Transmittance values of shape (num_rays, num_bins).
    
    Returns:
        torch.Tensor: Accumulated transmittance of shape (num_rays, num_bins).
    """
    accumulated_transmittance = torch.cumprod(betas, 1)
    return torch.cat((torch.ones(accumulated_transmittance.shape[0], 1, device=betas.device),
                     accumulated_transmittance[:, :-1]), dim=1)

def rendering(model, rays_o, rays_d, tn, tf, nb_bins=100, device='cpu', white_bckgr=True):
    """
    Render the color of rays using volume rendering.
    
    Args:
        model (torch.nn.Module): The NeRF model.
        rays_o (torch.Tensor): Ray origins of shape (num_rays, 3).
        rays_d (torch.Tensor): Ray directions of shape (num_rays, 3).
        tn (float): Near bound for ray sampling.
        tf (float): Far bound for ray sampling.
        nb_bins (int): Number of bins for sampling along the rays.
        device (str): Device for computation ('cpu' or 'cuda').
        white_bckgr (bool): Whether to assume a white background.
    
    Returns:
        torch.Tensor: Rendered colors of shape (num_rays, 3).
    """
    # Ensure inputs are on the correct device
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    model = model.to(device)
    
    # Sample points along the rays
    t = torch.linspace(tn, tf, nb_bins, device=device)  # [nb_bins]
    delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))  # [nb_bins]
    
    # Compute sampled points along the rays
    sampled_points = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1)  # [num_rays, nb_bins, 3]
   
    
    # Reshape points and directions for network input
    flat_points = sampled_points.reshape(-1, 3)
    expanded_dirs = rays_d.expand(nb_bins, rays_o.shape[0], 3).transpose(0, 1).reshape(-1, 3)
    
    # Predict densities and colors separately
    density = model.density(flat_points)
    colors = model.color(flat_points, expanded_dirs)
    # breakpoint()
    # Reshape outputs
    colors = colors.reshape((rays_o.shape[0], nb_bins, 3))  # [num_rays, nb_bins, 3]
    density = density.reshape((rays_o.shape[0], nb_bins))  # [num_rays, nb_bins]
    
    # Compute alpha and weights
    alpha = 1 - torch.exp(-density * delta.unsqueeze(0))  # [num_rays, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha) * alpha  # [num_rays, nb_bins]
   
    # Compute final colors
    if white_bckgr:
        c = (weights.unsqueeze(-1) * colors).sum(1)  # [num_rays, 3]
        # breakpoint()
        weight_sum = weights.sum(-1)  # [num_rays]
        return c + 1 - weight_sum.unsqueeze(-1)
    else:
        return (weights.unsqueeze(-1) * colors).sum(1)  # [num_rays, 3]
