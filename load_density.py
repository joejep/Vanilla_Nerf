# # import torch
# # from model2 import SplitNeRF

# # device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # # Load trained model checkpoint
# # checkpoint_path = "/home/eiyike/New_Vanilla_Nerf/nerf_model_epoch_3.pth"


# # checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# # # Define model with the same parameters used in training
# # model = SplitNeRF( Lpos=10, Ldir=4, hidden_dim=128).to(device)


# # # Load model weights
# # model.load_state_dict(checkpoint["model"])  
# # model.eval()  # Set to evaluation mode

# # # Define test query points (must be in [-bound, bound])
# # query_points =  (torch.rand(1024, 100, 3, dtype=torch.float32) * 4) - 2
# # query_points = query_points.reshape(-1, 3)

# # # Compute density
# # with torch.no_grad():
# #     result = model.density(query_points)

# # # Extract outputs
# # sigma = result["sigma"]  # Density values
# # geo_feat = result["geo_feat"]  # Geometric features

# # print("Sigma (Density):", sigma.shape)
# # print("Geometric Features:", geo_feat.shape)



# import torch
# from model2 import SplitNeRF

# # Device configuration
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # Load checkpoint and inspect its structure
# checkpoint_path = "/home/eiyike/New_Vanilla_Nerf/nerf_model_epoch_3.pth"
# checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# # Print checkpoint structure
# print("Checkpoint keys:", checkpoint.keys() if isinstance(checkpoint, dict) else "Checkpoint is not a dictionary")

# # Define model
# model = SplitNeRF(
#     Lpos=10,
#     Ldir=4,
#     hidden_dim=128
# ).to(device)

# # Load model weights - modify based on checkpoint structure
# if isinstance(checkpoint, dict):
#     if "model" in checkpoint:
#         model.load_state_dict(checkpoint["model"])
#     elif "state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["state_dict"])
#     else:
#         # If checkpoint is a direct state dict
#         model.load_state_dict(checkpoint)
# else:
#     # If checkpoint contains direct state dict
#     model.load_state_dict(checkpoint)

# model.eval()

# # Rest of your code remains the same
# query_points = (torch.rand(1024, 100, 3, dtype=torch.float32) * 4) - 2
# query_points = query_points.reshape(-1, 3)

# with torch.no_grad():
#     result = model.density(query_points)

# sigma = result["sigma"]
# geo_feat = result["geo_feat"]

# print("Sigma (Density):", sigma.shape)
# print("Geometric Features:", geo_feat.shape)



import torch
from model2 import SplitNeRF

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint_path = "/home/eiyike/New_Vanilla_Nerf/nerf_final_model.pth"
saved_model = torch.load(checkpoint_path, map_location=torch.device("cpu"))

# Create new model instance
model = SplitNeRF(
    Lpos=10,
    Ldir=4,
    hidden_dim=128
).to(device)

# Copy the state dict from the saved model
model.load_state_dict(saved_model.state_dict())

model.eval()

# Define test query points
query_points = (torch.rand(160000, 100, 3, dtype=torch.float32) * 4) - 2
query_points = query_points.reshape(-1, 3)

# Compute density
with torch.no_grad():
    result = model.density(query_points)

print("Sigma (Density):", result.shape)
# sigma = result["sigma"]
# geo_feat = result["geo_feat"]

# print("Sigma (Density):", sigma.shape)
# print("Geometric Features:", geo_feat.shape)
