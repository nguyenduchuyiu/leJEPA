import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.sigreg import SIGRegLoss

# --- MODULE: SPATIAL SOFTMAX (TRÁI TIM CỦA ROBOT VISION) ---
class SpatialSoftmax(nn.Module):
    def __init__(self, num_features, height, width, temperature=None):
        super().__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.temperature = nn.Parameter(torch.ones(1) * temperature) if temperature else 1.0

        # Tạo lưới tọa độ (pos_x, pos_y) cố định
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature_map):
        # feature_map: [B, C, H, W]
        B, C, H, W = feature_map.shape
        x = feature_map.view(B, C, -1) # [B, C, H*W]
        
        # Tính Softmax trên không gian 2D để tìm "tâm điểm" của sự chú ý
        softmax_attention = F.softmax(x / self.temperature, dim=-1)
        
        # Tính kỳ vọng tọa độ (Expected Coordinate)
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=2, keepdim=True)
        
        # Kết quả là tọa độ (x, y) của C keypoints -> [B, C*2]
        expected_xy = torch.cat([expected_x, expected_y], dim=2)
        return expected_xy.view(B, -1)

# --- MODULE: RESIDUAL MLP BLOCK ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.fc(x) # Skip connection

# --- MAIN MODEL ---
class LeJEPA_Robot(nn.Module):
    def __init__(self, action_dim=4, z_dim=128): # Tăng z_dim lên 128
        super().__init__()
        self.z_dim = z_dim
        import numpy as np # Import ở đây để SpatialSoftmax dùng

        # 1. ENCODER (CNN + Spatial Softmax)
        # Giả sử ảnh vào là 96x96
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # -> 48x48
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> 24x24
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> 12x12
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),# -> 12x12 (Giữ nguyên size để Spatial Softmax làm việc)
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        
        # Spatial Softmax sẽ biến 256 feature maps 12x12 thành 256 điểm tọa độ (x, y)
        # Output shape = 256 * 2 = 512
        self.spatial_softmax = SpatialSoftmax(256, 12, 12, temperature=0.1)
        
        # Project về z_dim
        self.projector = nn.Linear(512, z_dim)

        # 2. PREDICTOR (Residual MLP)
        # Mạnh hơn MLP thường gấp 10 lần
        self.predictor_net = nn.Sequential(
            nn.Linear(z_dim + action_dim, 256),
            nn.LayerNorm(256), nn.ReLU(),
            ResBlock(256), # Block 1
            ResBlock(256), # Block 2
            ResBlock(256), # Block 3
            nn.Linear(256, z_dim)
        )
        
        self.sigreg = SIGRegLoss(z_dim)

    def forward(self, obs, action, next_obs, lambda_coef=0.05): 
        # Encode
        h_t = self.conv_net(obs)
        spatial_t = self.spatial_softmax(h_t)
        z_t = self.projector(spatial_t)
        
        h_next = self.conv_net(next_obs)
        spatial_next = self.spatial_softmax(h_next)
        z_next_target = self.projector(spatial_next)
        
        # Predict
        z_next_pred = self.predictor_net(torch.cat([z_t, action], dim=1))
        
        # Loss
        loss_pred = F.mse_loss(z_next_pred, z_next_target)
        loss_reg = self.sigreg(z_t) + self.sigreg(z_next_target)
        total_loss = (1 - lambda_coef) * loss_pred + lambda_coef * loss_reg
        
        return total_loss, loss_pred, loss_reg
    
    # Hàm tiện ích cho Inference
    def encoder(self, obs):
        h = self.conv_net(obs)
        s = self.spatial_softmax(h)
        return self.projector(s)
        
    def predictor(self, z_and_action):
        return self.predictor_net(z_and_action)