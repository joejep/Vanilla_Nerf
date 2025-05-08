import torch
from model2 import SplitNeRF


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the entire model directly
model = torch.load("nerf_final_model.pth", map_location=device)
model = model.to(device)  # Move to the correct device

model.eval()

xyzs = (torch.rand(1024000, 3, dtype=torch.float32)) 
density2 = model.sigma_net(xyzs)


print("Density Output:", density2)
# xyzs = torch.randn(4096, 3).to(device) 
density = model.density(xyzs)


print("Density Output:", density.shape)  
breakpoint()
##### this is not required
# xyzs = xyzs.reshape(-1, 3) 
# ###############################################################  
# breakpoint()

