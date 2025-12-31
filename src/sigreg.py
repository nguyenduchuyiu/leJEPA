import torch
import torch.nn as nn

class SIGRegLoss(nn.Module):
    def __init__(self, z_dim, num_slices=1024, t_min=-5, t_max=5, num_points=17):
        super().__init__()
        self.z_dim = z_dim
        self.num_slices = num_slices
        
        # Tạo lưới tích phân t (Integration domain) theo bài báo
        # Từ -5 đến 5 là đủ để bao phủ Gaussian chuẩn [cite: 2627]
        self.register_buffer('t', torch.linspace(t_min, t_max, num_points))
        
        # Hàm đặc trưng lý thuyết của Gaussian chuẩn: exp(-0.5 * t^2) [cite: 2627]
        self.register_buffer('target_cf', torch.exp(-0.5 * self.t**2))

    def forward(self, z):
        """
        z: [Batch, z_dim] - Latent embeddings
        """
        N, D = z.shape
        
        # 1. Slice Sampling (Random Projections) [cite: 2614-2622]
        # Resampling mỗi step giúp đánh bại "Curse of Dimensionality" [cite: 2690]
        # Tạo ma trận chiếu ngẫu nhiên A
        A = torch.randn(D, self.num_slices, device=z.device)
        A = A / A.norm(p=2, dim=0, keepdim=True) # Normalize về unit sphere
        
        # 2. Chiếu z lên các hướng A -> [Batch, Slices]
        projections = z @ A 
        
        # 3. Tính Empirical Characteristic Function (ECF) [cite: 2628-2630]
        # ECF(t) = Mean( exp(i * t * proj) )
        # Ta cần tính cho mọi điểm t trong lưới tích phân
        # shape: [Batch, Slices, Num_points]
        args = projections.unsqueeze(-1) * self.t.unsqueeze(0).unsqueeze(0)
        
        # Dùng công thức Euler: exp(ix) = cos(x) + i*sin(x)
        # Tính mean trên Batch (dim 0) -> [Slices, Num_points]
        ecf_real = torch.cos(args).mean(dim=0)
        ecf_imag = torch.sin(args).mean(dim=0)
        
        # 4. Tính Loss so với Target Gaussian [cite: 2632]
        # Target CF là số thực (phần ảo = 0 vì Gaussian đối xứng)
        # Loss = |ECF - Target|^2 = (Real - Target)^2 + Imag^2
        diff_real = ecf_real - self.target_cf.unsqueeze(0)
        loss_vals = diff_real**2 + ecf_imag**2
        
        # Weighted L2 distance (Weight function w(t) = target_cf theo bài báo)
        weighted_loss = loss_vals * self.target_cf.unsqueeze(0)
        
        # 5. Tích phân (Trapezoidal rule) [cite: 2633]
        # Tổng hợp lỗi trên miền t
        total_loss = torch.trapz(weighted_loss, self.t, dim=1).mean()
        
        return total_loss