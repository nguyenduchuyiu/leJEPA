import os
import sys
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Allow running as a script: `python src/train.py`
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import LeJEPA_Robot
from config import COMMON, TRAIN

# --- CONFIG ---
BATCH_SIZE = TRAIN.batch_size      # [cite: 2797] LeJEPA chạy tốt với batch nhỏ >= 128
LEARNING_RATE = TRAIN.learning_rate  # [cite: 2879]
EPOCHS = TRAIN.epochs           # Train nhanh để test logic trước
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAMBDA = TRAIN.lambda_coef         # [cite: 2794] Recommended default
RANDOM_SHIFT_PX = getattr(TRAIN, "random_shift_px", 0)

# Multi-GPU (optional)
USE_DATAPARALLEL = TRAIN.use_dataparallel

DATA_DIR = COMMON.data_dir
NPZ_PATH = os.path.join(DATA_DIR, "train_buffer.npz")
META_PATH = os.path.join(DATA_DIR, "meta.npz")
OBS_PATH = os.path.join(DATA_DIR, "obs.npy")
ACTIONS_PATH = os.path.join(DATA_DIR, "actions.npy")
NEXT_OBS_PATH = os.path.join(DATA_DIR, "next_obs.npy")


class MemmapTransitionDataset(Dataset):
    def __init__(self, obs_mm: np.ndarray, actions_mm: np.ndarray, next_obs_mm: np.ndarray):
        if len(obs_mm) != len(actions_mm) or len(obs_mm) != len(next_obs_mm):
            raise ValueError(f"Length mismatch: obs={len(obs_mm)} actions={len(actions_mm)} next_obs={len(next_obs_mm)}")
        self.obs_mm = obs_mm
        self.actions_mm = actions_mm
        self.next_obs_mm = next_obs_mm

    def __len__(self):
        return len(self.obs_mm)

    def __getitem__(self, idx):
        # obs/next_obs are uint8 CHW; actions are float32
        # Memmap slices are often non-writable; copy to avoid PyTorch warning/UB.
        obs = torch.from_numpy(np.array(self.obs_mm[idx], copy=True)).float().div_(255.0)
        act = torch.from_numpy(np.array(self.actions_mm[idx], copy=True)).float()
        next_obs = torch.from_numpy(np.array(self.next_obs_mm[idx], copy=True)).float().div_(255.0)
        return obs, act, next_obs


def load_buffer(data_dir: str = DATA_DIR, fraction: float = 0.25):
    """
    Supports 2 formats produced by 01_collect_data.py:
    - STREAM_TO_NPY=True: obs.npy/actions.npy/next_obs.npy + meta.npz (preferred, low RAM)
    - STREAM_TO_NPY=False: train_buffer.npz with keys obs/actions/next_obs
    Only loads a given fraction of the data.
    """
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    npz_path = os.path.join(data_dir, "train_buffer.npz")
    meta_path = os.path.join(data_dir, "meta.npz")
    obs_path = os.path.join(data_dir, "obs.npy")
    actions_path = os.path.join(data_dir, "actions.npy")
    next_obs_path = os.path.join(data_dir, "next_obs.npy")

    if os.path.exists(meta_path) and os.path.exists(obs_path) and os.path.exists(actions_path) and os.path.exists(next_obs_path):
        meta = np.load(meta_path)
        n_full = int(meta["n"][0])
        n = max(1, min(n_full, int(n_full * fraction)))
        obs_mm = np.load(obs_path, mmap_mode="r")[:n]
        actions_mm = np.load(actions_path, mmap_mode="r")[:n]
        next_obs_mm = np.load(next_obs_path, mmap_mode="r")[:n]
        return MemmapTransitionDataset(obs_mm, actions_mm, next_obs_mm)

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        total = data["obs"].shape[0]
        n = max(1, min(total, int(total * fraction)))
        obs = torch.from_numpy(data["obs"][:n]).float().div_(255.0)
        actions = torch.from_numpy(data["actions"][:n]).float()
        next_obs = torch.from_numpy(data["next_obs"][:n]).float().div_(255.0)
        return TensorDataset(obs, actions, next_obs)

    raise FileNotFoundError(
        f"Could not find dataset in {data_dir}. Expected either:\n"
        f"- {meta_path} + {obs_path} + {actions_path} + {next_obs_path}\n"
        f"- or {npz_path}"
    )

def random_shift_pair(x: torch.Tensor, y: torch.Tensor, pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Random shift (translation) by up to +/- pad pixels, applied with the SAME offsets to x and y.
    Uses replicate padding + random crop (simple, no extra deps).
    """
    if pad <= 0:
        return x, y
    if x.shape != y.shape or x.dim() != 4:
        raise ValueError(f"Expected x,y same shape [B,C,H,W], got x={tuple(x.shape)} y={tuple(y.shape)}")
    b, c, h, w = x.shape
    x_p = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    y_p = F.pad(y, (pad, pad, pad, pad), mode="replicate")
    off_y = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    off_x = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    x_out = torch.empty_like(x)
    y_out = torch.empty_like(y)
    for i in range(b):
        oy = int(off_y[i])
        ox = int(off_x[i])
        x_out[i] = x_p[i, :, oy:oy + h, ox:ox + w]
        y_out[i] = y_p[i, :, oy:oy + h, ox:ox + w]
    return x_out, y_out


def main():
    # 1. Load Data
    print("Loading data...")
    dataset = load_buffer(DATA_DIR, fraction=1.0)

    # Tạo DataLoader
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

    # 2. Init Model
    model = LeJEPA_Robot(action_dim=4, z_dim=64).to(DEVICE)
    if USE_DATAPARALLEL and DEVICE == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # [cite: 2879]

    # 3. Training Loop
    print(f"Start Training LeJEPA on {DEVICE}...")
    history = {'loss': [], 'pred': [], 'reg': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss_epoch = 0

        loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for b_obs, b_act, b_next in loader_tqdm:
            b_obs, b_act, b_next = b_obs.to(DEVICE), b_act.to(DEVICE), b_next.to(DEVICE)

            optimizer.zero_grad()

            # RandomShift augmentation (same shift for obs & next_obs)
            if RANDOM_SHIFT_PX and RANDOM_SHIFT_PX > 0:
                with torch.no_grad():
                    b_obs, b_next = random_shift_pair(b_obs, b_next, pad=int(RANDOM_SHIFT_PX))

            loss, l_pred, l_reg = model(b_obs, b_act, b_next, lambda_coef=LAMBDA)

            # DataParallel gathers per-GPU scalars into a 1D tensor; reduce to a scalar for backward/logging.
            if loss.dim() != 0:
                loss = loss.mean()
            if l_pred.dim() != 0:
                l_pred = l_pred.mean()
            if l_reg.dim() != 0:
                l_reg = l_reg.mean()

            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            loader_tqdm.set_postfix({"Loss": f"{loss.item():.4f}", "Pred": f"{l_pred.item():.4f}", "Reg": f"{l_reg.item():.4f}"})

        avg_loss = total_loss_epoch / len(loader)
        history['loss'].append(avg_loss)
        history['pred'].append(l_pred.item()) # Log batch cuối để tham khảo
        history['reg'].append(l_reg.item())

        # print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Pred: {l_pred.item():.6f} | Reg: {l_reg.item():.6f}")

    # 4. Save Model
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, COMMON.model_path)
    print("Model saved!")

    # (Optional) Visualize Loss Curve
    plt.plot(history['loss'], label='Total Loss')
    plt.legend()
    plt.title("LeJEPA Training Progress")
    plt.savefig("loss_curve.png")
    plt.show()


if __name__ == "__main__":
    main()