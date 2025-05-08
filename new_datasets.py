# import json
# import numpy as np
# import imageio.v3 as iio
# from skimage.transform import resize
# import os
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class data_preprocessing:
#     def __init__(self, data_path='', mode='', target_size=(800, 800)):
#         """
#         Initialize the data preprocessing pipeline.

#         Args:
#             data_path (str): Path to the dataset directory.
#             mode (str): Dataset mode (e.g., 'train', 'val', 'test').
#             target_size (tuple): Target image size (width, height).
#         """
#         self.data_path = data_path
#         self.mode = mode
#         self.json_name = f'transforms_{mode}.json'
#         self.img_wh = target_size
#         self.images = []
#         self.c2ws = []
#         self.N = 0

#         # Load JSON data
#         self.jsonPath = os.path.join(self.data_path, self.json_name)
#         if not os.path.exists(self.jsonPath):
#             raise FileNotFoundError(f"JSON file not found: {self.jsonPath}")

#         with open(self.jsonPath, 'r') as file:
#             data = json.load(file)

#         self.fieldOfView = data["camera_angle_x"]
#         self.Frames = sorted(data["frames"], key=lambda d: d['file_path'])

#         # Calculate focal length
#         original_width = 800  # Adjust based on dataset
#         self.focal = 0.5 * original_width / np.tan(0.5 * self.fieldOfView)
#         self.focal = self.focal * self.img_wh[0] / original_width
#         logger.info(f"Focal length: {self.focal}")

#         # Process each frame
#         for frame in self.Frames:
#             image_filename = frame["file_path"].split('/')[-1] + '.png'
#             imagePath = os.path.join(self.data_path, mode, image_filename)
            
#             if not os.path.exists(imagePath):
#                 raise FileNotFoundError(f"Image file not found: {imagePath}")
            
#             img = iio.imread(imagePath) / 255.
#             img = resize(img, self.img_wh)
#             self.images.append(img[None, ...])
#             self.c2ws.append(frame["transform_matrix"])

#         # Convert lists to numpy arrays
#         self.images = np.concatenate(self.images)
#         self.c2ws = np.array(self.c2ws)
#         self.N = self.images.shape[0]
#         logger.info(f"Loaded {self.N} images of shape {self.images.shape}")

#         # Handle alpha channel if present
#         if self.images.shape[3] == 4:
#             self.images = self.images[..., :3] * self.images[..., -1:] + (1 - self.images[..., -1:])
#         logger.info(f"Final images shape: {self.images.shape}")

#     def get_rays(self):
#         """
#         Generate rays for all images.

#         Returns:
#             rays_o (np.ndarray): Ray origins of shape (N, H*W, 3).
#             rays_d (np.ndarray): Ray directions of shape (N, H*W, 3).
#             target_px_values (np.ndarray): Target pixel values of shape (N, H*W, 3).
#             N (int): Number of images.
#         """
#         logger.info(f"Generating rays for {self.N} images of size {self.img_wh[1]} x {self.img_wh[0]}")

#         rays_o = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
#         rays_d = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
#         target_px_values = self.images.reshape((self.N, self.img_wh[1] * self.img_wh[0], 3))
 
#         # Precompute u and v
#         u = np.arange(self.img_wh[0])
#         v = np.arange(self.img_wh[1])
#         u, v = np.meshgrid(u, v)
#         dirs = np.stack((u - self.img_wh[0] / 2, -(v - self.img_wh[1] / 2), -np.ones_like(u) * self.focal), axis=-1)

#         for i in range(self.N):
#             c2w = self.c2ws[i]
#             dirs_transformed = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
#             dirs_transformed = dirs_transformed / np.linalg.norm(dirs_transformed, axis=-1, keepdims=True)
#             rays_d[i] = dirs_transformed.reshape(-1, 3)
#             rays_o[i] += c2w[:3, -1]

#         logger.info("Ray generation complete:")
#         logger.info(f"Ray origins shape: {rays_o.shape}")
#         logger.info(f"Ray directions shape: {rays_d.shape}")
#         logger.info(f"Target pixel values shape: {target_px_values.shape}")

#         return rays_o, rays_d, target_px_values, self.N


import json
import numpy as np
import imageio.v3 as iio
from skimage.transform import resize
import os

class data_preprocessing:
    def __init__(self, data_path='', mode='', target_size=(800, 800)):
        self.data_path = data_path
        self.mode = mode
        self.json_name = f'transforms_{mode}.json'
        self.img_wh = target_size
        self.images = []
        self.c2ws = []
        self.N = 0
        
        # Create proper paths
        self.jsonPath = os.path.join(self.data_path, self.json_name)
        
        # Load JSON data
        with open(self.jsonPath, 'r') as file:
            data = json.load(file)

        self.fieldOfView = data["camera_angle_x"]
        self.Frames = data["frames"]
        self.Frames = sorted(self.Frames, key=lambda d: d['file_path'])

        # Calculate focal length
        self.focal = 0.5 * 800 / np.tan(0.5 * self.fieldOfView)  # original focal length
        self.focal = self.focal * self.img_wh[0] / 800  # modify focal length to match size self.img_wh
        print("focal_length: ", self.focal)

        # Process each frame
        for frame in self.Frames:
            # Construct proper image path
            image_filename = frame["file_path"].split('/')[-1] + '.png'
            imagePath = os.path.join(self.data_path, mode, image_filename)
            
            # Verify file exists
            if not os.path.exists(imagePath):
                raise FileNotFoundError(f"Image file not found: {imagePath}")
            
            # Read and process image
            img = iio.imread(imagePath) / 255.
            img = resize(img, self.img_wh)
            self.images.append(img[None, ...])
            # Get camera-to-world transform
            c2w = frame["transform_matrix"]
            self.c2ws.append(c2w)

        # Convert lists to numpy arrays
        self.images = np.concatenate(self.images)
        self.c2ws = np.array(self.c2ws)

        self.N = self.images.shape[0]
        print("Images shape: ", self.images.shape)

        # Handle alpha channel if present
        if self.images.shape[3] == 4:
            self.images = self.images[..., :3] * self.images[..., -1:] + (1 - self.images[..., -1:])
        print("Final images shape: ", self.images.shape)

    def get_rays(self):
        print(f"Generating rays for {self.N} images of size {self.img_wh[1]} x {self.img_wh[0]}")

        rays_o = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
        rays_d = np.zeros((self.N, self.img_wh[1] * self.img_wh[0], 3))
        target_px_values = self.images.reshape((self.N, self.img_wh[1] * self.img_wh[0], 3))

        for i in range(self.N):
            c2w = self.c2ws[i]
            f = self.focal

            u = np.arange(self.img_wh[0])
            v = np.arange(self.img_wh[1])

            u, v = np.meshgrid(u, v)
            dirs = np.stack((u - self.img_wh[0] / 2, -(v - self.img_wh[1] / 2), -np.ones_like(u) * f), axis=-1)
            
            dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
            rays_d[i] = dirs.reshape(-1, 3)
            rays_o[i] += c2w[:3, -1]
            

        print("Ray generation complete:")
        print('Ray origins shape:', rays_o.shape)
        print('Ray directions shape:', rays_d.shape)
        print('Target pixel values shape:', target_px_values.shape)

        return rays_o, rays_d, target_px_values, self.N