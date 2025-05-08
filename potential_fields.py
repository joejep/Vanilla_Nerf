# import json
# import cv2
# import torch
# from packaging import version as pver
# import matplotlib.pyplot as plt
# # from nerf.network_tcnn import NeRFNetwork
# # from network_tcnn import NeRFNetwork  # Ensure this is correctly imported
# # from activation import trunc_exp  # If needed

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# json_path = "/home/eiyike/DATA/Potential_fields_dataset/FINAL_DATA/transforms_train.json"
# data_dir = "/home/eiyike/DATA/Potential_fields_dataset/FINAL_DATA"

# def load_camera_data(json_path, data_dir):
#     with open(json_path, "r") as f:
#         camera_data = json.load(f)

#     fl_x = camera_data["fl_x"]
#     fl_y = camera_data["fl_y"]
#     cx = camera_data["cx"]
#     cy = camera_data["cy"]
#     W = int(camera_data["w"])
#     H = int(camera_data["h"])

#     intrinsic_matrix = torch.tensor([
#         [fl_x, 0, cx],
#         [0, fl_y, cy],
#         [0, 0, 1]
#     ], dtype=torch.float32)

#     frames = camera_data["frames"]
#     frames = sorted(frames, key=lambda d: d['file_path'])
#     images = []
#     poses = []

#     for frame in frames:
#         file_path = frame["file_path"]
#         image = cv2.imread(f"{data_dir}/{file_path}.png")
#         if image is not None:
      
#             image = torch.from_numpy(image).float() / 255.0  #  0, 1
#             images.append(image)

        
#             transform_matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
#             poses.append(transform_matrix)
#         else:
#             print(f"Warning: Image at {file_path} could not be loaded.")
#     poses_array = torch.stack(poses)
#     intrinsics = torch.tensor([fl_x, fl_y, cx, cy], dtype=torch.float32)
#     images = torch.stack(images)
#     N= images.shape[0]

#     return poses_array, images, intrinsics, W, H,N

# poses, images, intrinsics, W, H,N= load_camera_data(json_path, data_dir)



# def get_rays(self):
#         print(f"Generating rays for {self.N} images of size {self.img_wh[1]} x {self.img_wh[0]}")

#         rays_o = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
#         rays_d = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
#         target_px_values = self.images.reshape((self.N, self.img_wh[1] * self.img_wh[0], 3))

#         for i in range(self.N):
#             c2w = self.c2ws[i]
#             f = self.focal

#             u = np.arange(self.img_wh[0])
#             v = np.arange(self.img_wh[1])

#             u, v = np.meshgrid(u, v)
#             dirs = np.stack((u - self.img_wh[0] / 2, -(v - self.img_wh[1] / 2), -np.ones_like(u) * f), axis=-1)
            
#             dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
#             dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
#             rays_d[i] = dirs.reshape(-1, 3)
#             rays_o[i] += c2w[:3, -1]
            

#         print("Ray generation complete:")
#         print('Ray origins shape:', rays_o.shape)
#         print('Ray directions shape:', rays_d.shape)
#         print('Target pixel values shape:', target_px_values.shape)

#         return rays_o, rays_d, target_px_values, self.N



import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tn = 2
tf = 8
nb_epochs = 5
lr = 1e-3
gamma = 0.5
nb_bins = 100
batch_size=1024

json_path = "/home/eiyike/3D_RECONSTRUCTION/VANILLA_NERF/vanilla_dataset/transforms_train.json"
data_dir = "/home/eiyike/3D_RECONSTRUCTION/VANILLA_NERF/vanilla_dataset/"

def load_camera_data(json_path, data_dir):
    with open(json_path, "r") as f:
        camera_data = json.load(f)

    fl_x = camera_data["fl_x"]
    fl_y = camera_data["fl_y"]
    cx = camera_data["cx"]
    cy = camera_data["cy"]
    W = int(camera_data["w"])
    H = int(camera_data["h"])

    intrinsic_matrix = torch.tensor([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ], dtype=torch.float32)

    frames = camera_data["frames"]
    frames = sorted(frames, key=lambda d: d['file_path'])
    images = []
    poses = []

    for frame in frames:
        file_path = frame["file_path"]
        image = cv2.imread(f"{data_dir}/{file_path}.png")
        if image is not None:
            image = torch.from_numpy(image).float() / 255.0  # Normalize to [0, 1]
            images.append(image)

            transform_matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            poses.append(transform_matrix)
        else:
            print(f"Warning: Image at {file_path} could not be loaded.")
    poses_array = torch.stack(poses)
    intrinsics = torch.tensor([fl_x, fl_y, cx, cy], dtype=torch.float32)
    images = torch.stack(images)
    N = images.shape[0]

    return poses_array, images, intrinsics, W, H, N

poses, images, intrinsics, W, H, N = load_camera_data(json_path, data_dir)

def get_rays(poses, images, intrinsics, W, H, N):
    print(f"Generating rays for {N} images of size {H} x {W}")

    # Ensure all tensors are on the same device
    device = poses.device  # Use the device of the poses tensor
    fl_x, fl_y, cx, cy = intrinsics.to(device)
    rays_o = torch.zeros((N, H * W, 3), device=device)
    rays_d = torch.zeros((N, H * W, 3), device=device)
    target_px_values = images.reshape((N, H * W, 3)).to(device)

    for i in range(N):
        c2w = poses[i]  # Already on the correct device
        f = torch.tensor([fl_x, fl_y], device=device)

        u = torch.arange(W, device=device)
        v = torch.arange(H, device=device)

        u, v = torch.meshgrid(u, v, indexing='xy')  # Add indexing='xy' to avoid warning
        dirs = torch.stack(((u - cx) / fl_x, -(v - cy) / fl_y, -torch.ones_like(u)), dim=-1)
        
        # Ensure dirs is on the same device as c2w
        dirs = dirs.to(device)
        dirs = torch.matmul(c2w[:3, :3], dirs.unsqueeze(-1)).squeeze(-1)
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] += c2w[:3, -1]

    print("Ray generation complete:")
    print('Ray origins shape:', rays_o.shape)
    print('Ray directions shape:', rays_d.shape)
    print('Target pixel values shape:', target_px_values.shape)

    return rays_o, rays_d, target_px_values, N
rays_o, rays_d, target_px_values, N = get_rays(poses, images, intrinsics, W, H, N)


# Full data loader
dataloader = DataLoader(
    torch.cat(
        (torch.from_numpy(rays_o).reshape(-1, 3).type(torch.float),
        torch.from_numpy(rays_d).reshape(-1, 3).type(torch.float),
        torch.from_numpy(target_px_values).reshape(-1, 3).type(torch.float)),
        dim=1),
    batch_size=batch_size, shuffle=True
)


for batch in tqdm(dataloader):
            ray_origins = batch[:, :3].to(device)  # Ray origins
            
            ray_directions = batch[:, 3:6].to(device)  # Ray directions
            target = batch[:, 6:].to(device)  # Target pixel values
            breakpoint()


t = torch.linspace(tn, tf, nb_bins, device=device)  # [nb_bins]
delta = torch.cat((t[1:] - t[:-1], torch.tensor([1e10], device=device)))  # [nb_bins]

# Compute sampled points along the rays
sampled_points = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1)  # [num_rays, nb_bins, 3]


# Reshape points and directions for network input
flat_points = sampled_points.reshape(-1, 3)
expanded_dirs = rays_d.expand(nb_bins, rays_o.shape[0], 3).transpose(0, 1).reshape(-1, 3)


