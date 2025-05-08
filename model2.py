

# import torch
# import torch.nn as nn
# import tinycudann as tcnn

# class PositionalEncoder:
#     def __init__(self, L):
#         self.L = L
        
#     def encode(self, x):
#         out = [x]
#         for j in range(self.L):
#             out.append(torch.sin(2 ** j * x))
#             out.append(torch.cos(2 ** j * x))
#         return torch.cat(out, dim=1)

# class SigmaNetwork(nn.Module):
#     def __init__(self, Lpos=10, hidden_dim=256):
#         super(SigmaNetwork, self).__init__()
        
#         self.positional_encoder = PositionalEncoder(Lpos)
#         input_dim = Lpos * 6 + 3  # 3 for xyz
        
#         self.block1 = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
#         )
        
#         self.block2 = nn.Sequential(
#             nn.Linear(hidden_dim + input_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#     def forward(self, xyz):
#         # Ensure input is on the same device as the model
#         xyz = xyz.to(self.block1[0].weight.device)
#         x_emb = self.positional_encoder.encode(xyz)
#         h = self.block1(x_emb)
#         h = self.block2(torch.cat((h, x_emb), dim=1))
#         return torch.relu(h.squeeze(-1))

# class ColorNetwork(nn.Module):
#     def __init__(self, Lpos=10, Ldir=4, hidden_dim=256):
#         super(ColorNetwork, self).__init__()
        
#         self.positional_encoder_pos = PositionalEncoder(Lpos)
#         self.positional_encoder_dir = PositionalEncoder(Ldir)
        
#         input_pos_dim = Lpos * 6 + 3  # 3 for xyz
#         input_dir_dim = Ldir * 6 + 3  # 3 for direction
        
#         self.block1 = nn.Sequential(
#             nn.Linear(input_pos_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
#         )
        
#         self.block2 = nn.Sequential(
#             nn.Linear(hidden_dim + input_pos_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
        
#         self.rgb_head = nn.Sequential(
#             nn.Linear(hidden_dim + input_dir_dim, hidden_dim // 2), nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 3),
#             nn.Sigmoid()
#         )
        
#     def forward(self, xyz, d):
#         # Ensure inputs are on the same device as the model
#         device = self.block1[0].weight.device
#         xyz = xyz.to(device)
#         d = d.to(device)
        
#         x_emb = self.positional_encoder_pos.encode(xyz)
#         d_emb = self.positional_encoder_dir.encode(d)
        
#         h = self.block1(x_emb)
#         h = self.block2(torch.cat((h, x_emb), dim=1))
#         c = self.rgb_head(torch.cat((h, d_emb), dim=1))
#         return c

# class SplitNeRF(nn.Module):
#     def __init__(self, Lpos=10, Ldir=4, hidden_dim=256):
#         super(SplitNeRF, self).__init__()
#         self.sigma_net = SigmaNetwork(Lpos, hidden_dim)
#         self.color_net = ColorNetwork(Lpos, Ldir, hidden_dim)
        
#     def to(self, device):
#         super().to(device)
#         self.sigma_net = self.sigma_net.to(device)
#         self.color_net = self.color_net.to(device)
#         return self
        
#     def forward(self, xyz, d):
#         sigma = self.density(xyz)
#         breakpoint()
#         color = self.color(xyz, d)
        
#         return color, sigma
    
#     def density(self, xyz):
#         """Compute the density at given 3D positions."""
        
#         return self.sigma_net(xyz)
    
#     def color(self, xyz, d):
#         """Compute the color at given 3D positions and view directions."""
#         return self.color_net(xyz, d)
    
#     def intersect(self, xyz, d):
#         """Legacy method for compatibility - calls forward()"""
#         return self.forward(xyz, d)


#FULLYFUSED



import torch
import torch.nn as nn
import tinycudann as tcnn

class PositionalEncoder:
    def __init__(self, L):
        self.L = L
        
    def encode(self, x):
        out = [x]
        for j in range(self.L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

class SigmaNetwork(nn.Module):
    def __init__(self, Lpos=10, hidden_dim=128):
        super(SigmaNetwork, self).__init__()
        
        self.positional_encoder = PositionalEncoder(Lpos)
        input_dim = Lpos * 6 + 3  # 3 for xyz
        
        # Replace sequential blocks with FullyFusedMLP
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": 8  # Combined layers from block1 and block2
        }
        
        self.network = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=1,
            network_config=network_config
        )
        
    def forward(self, xyz):
        xyz = xyz.to(self.network.params.device)
        x_emb = self.positional_encoder.encode(xyz)
        sigma = self.network(x_emb)
        return torch.relu(sigma.squeeze(-1))

class ColorNetwork(nn.Module):
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=128):
        super(ColorNetwork, self).__init__()
        
        self.positional_encoder_pos = PositionalEncoder(Lpos)
        self.positional_encoder_dir = PositionalEncoder(Ldir)
        
        input_pos_dim = Lpos * 6 + 3  # 3 for xyz
        input_dir_dim = Ldir * 6 + 3  # 3 for direction
        total_input_dim = input_pos_dim + input_dir_dim
        
        # Single FullyFusedMLP for the entire color network
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",  # Final activation for RGB
            "n_neurons": hidden_dim,
            "n_hidden_layers": 8  # Combined layers from all blocks
        }
        
        self.network = tcnn.Network(
            n_input_dims=total_input_dim,
            n_output_dims=3,  # RGB output
            network_config=network_config
        )
        
    def forward(self, xyz, d):
        device = self.network.params.device
        xyz = xyz.to(device)
        d = d.to(device)
        
        x_emb = self.positional_encoder_pos.encode(xyz)
        d_emb = self.positional_encoder_dir.encode(d)
        
        # Concatenate all inputs
        network_input = torch.cat([x_emb, d_emb], dim=1)
        return self.network(network_input)

class SplitNeRF(nn.Module):
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=128):
        super(SplitNeRF, self).__init__()
        self.sigma_net = SigmaNetwork(Lpos, hidden_dim)
        self.color_net = ColorNetwork(Lpos, Ldir, hidden_dim)
        
    def to(self, device):
        # Note: tcnn networks automatically handle device placement
        super().to(device)
        return self
        
    def forward(self, xyz, d):
        sigma = self.density(xyz)
        color = self.color(xyz, d)
        return color, sigma
    
    def density(self, xyz):
        """Compute the density at given 3D positions."""
        return self.sigma_net(xyz)
    
    def color(self, xyz, d):
        """Compute the color at given 3D positions and view directions."""
        return self.color_net(xyz, d)
    
    def intersect(self, xyz, d):
        """Legacy method for compatibility - calls forward()"""
        return self.forward(xyz, d)