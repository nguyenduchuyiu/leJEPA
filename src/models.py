import torch.nn.functional as F
import torch.nn as nn
from src.sigreg import SIGRegLoss
import torch 

class LeJEPA_Robot(nn.Module):
    def __init__(self, action_dim=4, z_dim=64):
        super().__init__()
        self.z_dim = z_dim
        
        # --- Encoder: Image (3, 64, 64) -> z (64) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> 4x4
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, z_dim),
            # KHÔNG CẦN BatchNorm cuối cùng vì SIGReg sẽ lo việc chuẩn hóa
        )
        
        # --- Predictor: (z, action) -> z_next ---
        self.predictor = nn.Sequential(
            nn.Linear(z_dim + action_dim, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, z_dim) # Output dự đoán z tiếp theo
        )
        
        # Loss components
        self.sigreg = SIGRegLoss(z_dim)

    def forward(self, obs, action, next_obs, lambda_coef=0.05): 
        """
        obs: [B, 3, 64, 64]
        action: [B, 4]
        next_obs: [B, 3, 64, 64]
        lambda_coef: Hệ số cân bằng (Bài báo recommend 0.05 [cite: 2794])
        """
        # 1. Encode
        z_t = self.encoder(obs)
        z_next_target = self.encoder(next_obs) # Target embedding
        
        # 2. Predict Dynamics
        # Input: Latent hiện tại + Action
        z_next_pred = self.predictor(torch.cat([z_t, action], dim=1))
        
        # 3. Prediction Loss (MSE) [cite: 2739]
        # Học vật lý: Nếu ở z_t làm a_t thì sẽ đến z_next_target
        loss_pred = F.mse_loss(z_next_pred, z_next_target)
        
        # 4. SIGReg Loss (Regularization) [cite: 2746]
        # Ép không gian latent không bị sập (collapse) về 0
        # Áp dụng cho cả z_t và z_next_target để đảm bảo toàn không gian đẹp
        loss_reg = self.sigreg(z_t) + self.sigreg(z_next_target)
        
        # Tổng Loss: (1 - lambda) * Pred + lambda * Reg
        # Lưu ý: Bài báo dùng công thức trung bình view, ở đây ta giản lược
        total_loss = (1 - lambda_coef) * loss_pred + lambda_coef * loss_reg
        
        return total_loss, loss_pred, loss_reg