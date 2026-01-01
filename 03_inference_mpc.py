import os
from config import COMMON, INFER
os.environ.setdefault("MUJOCO_GL", COMMON.mujoco_gl)
import torch
import numpy as np
import metaworld
import cv2
import mujoco
import random
import torch.nn.functional as F
from src.models import LeJEPA_Robot
# --- CONFIG ---
TASK_NAME = COMMON.task_name
IMG_SIZE = COMMON.img_size
RENDER_SIZE = COMMON.render_size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = COMMON.model_path
CAMERA_NAME = COMMON.camera_name

# Multi-GPU (optional). In this script we call encoder/predictor directly,
# so wrap those modules (not the whole model) to actually use DataParallel.
USE_DATAPARALLEL = INFER.use_dataparallel

# Visual debugging (giống 01_collect_data.py)
PREVIEW = INFER.preview
PREVIEW_WAIT_MS = INFER.preview_wait_ms
PREVIEW_SCALE = INFER.preview_scale

# Config MPPI (Bộ não quy hoạch)
HORIZON = INFER.horizon
NUM_SAMPLES = INFER.num_samples
TEMPERATURE = INFER.temperature

# Reset/episode control (giống 01_collect_data.py)
NUM_EPISODES = INFER.num_episodes
MAX_STEPS = INFER.max_steps

def get_env():
    ml1 = metaworld.ML1(TASK_NAME)
    env = ml1.train_classes[TASK_NAME](render_mode="rgb_array")
    return env, ml1

def chw_rgb_to_hwc_bgr(img_chw: np.ndarray) -> np.ndarray:
    """Convert CHW RGB uint8 -> HWC BGR uint8 for OpenCV display/video."""
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    return cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

def grab_frame_np(renderer, data, cam_name):
    """Return CHW uint8 RGB (after flip + resize)."""
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    img = np.flipud(img)
    img = cv2.resize(img, IMG_SIZE)
    img = np.transpose(img, (2, 0, 1))
    return img.astype(np.uint8, copy=False)

def grab_frame(renderer, data, cam_name):
    img = grab_frame_np(renderer, data, cam_name)
    return torch.tensor(img).float().to(DEVICE) / 255.0

# --- HÀM QUAN TRỌNG: LẤY ẢNH GOAL ---
def capture_goal_image(env, renderer, cam_name):
    """
    Hack: Dịch chuyển vật đến goal, chụp ảnh.
    Index chuẩn cho Sawyer Push V3:
    - 0-6: Robot Arm
    - 7-8: Gripper (Right/Left finger)
    - 9-11: Object Position (X, Y, Z) <--- MỤC TIÊU CỦA TA Ở ĐÂY
    """
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Backup trạng thái cũ
    old_qpos = data.qpos.copy()
    old_qvel = data.qvel.copy()

    # Lấy vị trí goal
    obs_dict = env.unwrapped._get_obs_dict()
    goal_pos = obs_dict['state_desired_goal']

    # --- SỬA LỖI: GHI VÀO INDEX 9:12 (Thay vì 7:10) ---
    data.qpos[9:12] = goal_pos
    data.qvel[9:12] = 0.0
    
    mujoco.mj_forward(model, data)

    # --- BƯỚC 2: TÀNG HÌNH CHI THUẬT (WHITELIST VERSION) ---
    old_rgba = model.geom_rgba.copy()
    old_site_rgba = model.site_rgba.copy()
    
    # Những thứ BẮT BUỘC PHẢI GIỮ LẠI (Case-insensitive)
    # world: Bầu trời/nền
    # table: Cái bàn
    # wall: Tường chắn
    # obj: Vật thể cần đẩy
    # puck: Tên khác của vật thể
    # goal: Điểm đích (nếu có visual)
    KEEP_KEYWORDS = ['world', 'table', 'wall', 'obj', 'puck', 'goal']

    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        if body_name:
            # Chuyển về chữ thường để so sánh cho dễ
            name_lower = body_name.lower()
            
            # Logic: Nếu tên body KHÔNG chứa từ khóa an toàn -> ẨN NGAY
            is_safe = any(safe_key in name_lower for safe_key in KEEP_KEYWORDS)
            
            if not is_safe:
                # Đây là robot hoặc rác -> Hide
                for geom_id in range(model.ngeom):
                    if model.geom_bodyid[geom_id] == body_id:
                        model.geom_rgba[geom_id, 3] = 0.0 # Alpha = 0 (Tàng hình)
    
    # 2.2. ẨN SITE (Mấy cái đầu đỏ/Cảm biến) <--- FIX Ở ĐÂY
    for site_id in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        body_id = model.site_bodyid[site_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        
        # Check xem Site này có thuộc về body "an toàn" không
        # Hoặc check trực tiếp tên site
        full_name = (str(site_name) + str(body_name)).lower()
        is_safe = any(safe_key in full_name for safe_key in KEEP_KEYWORDS)
        
        if not is_safe:
            model.site_rgba[site_id, 3] = 0.0 # Tàng hình nốt
    
    # --- BƯỚC 3: CHỤP ẢNH ---
    # Cần update scene thì thay đổi màu sắc mới có hiệu lực
    renderer.update_scene(data, camera=cam_name)
    # Chụp ảnh goal
    img_goal_chw = grab_frame_np(renderer, data, cam_name)
    img_tensor = torch.tensor(img_goal_chw).float().to(DEVICE) / 255.0

    # Trả lại màu sắc (Hiện hình)
    model.geom_rgba[:] = old_rgba
    model.site_rgba[:] = old_site_rgba # Trả lại màu cho Site

    # Restore trạng thái cũ
    data.qpos[:] = old_qpos
    data.qvel[:] = old_qvel
    mujoco.mj_forward(model, data)

    return img_tensor.unsqueeze(0), img_goal_chw  # [1, 3, 64, 64], CHW uint8

# --- MPPI CONTROLLER ---
def plan_with_mppi(model, z_curr, z_goal):
    """
    Tưởng tượng ra hàng nghìn viễn cảnh và chọn hành động tốt nhất.
    """
    B, D = z_curr.shape
    
    # 1. Random ra chuỗi hành động [Num_samples, Horizon, Action_dim]
    # Action space [-1, 1]
    noise = torch.randn(NUM_SAMPLES, HORIZON, 4, device=DEVICE)
    
    # Action đề xuất (có thể dùng mean của bước trước, ở đây dùng pure noise cho đơn giản)
    actions = torch.tanh(noise) # Ép về [-1, 1]
    
    # 2. Latent Rollout (Tưởng tượng)
    z_states = z_curr.repeat(NUM_SAMPLES, 1) # Bắt đầu từ trạng thái hiện tại
    cumulative_rewards = torch.zeros(NUM_SAMPLES, device=DEVICE)
    
    for t in range(HORIZON):
        act_t = actions[:, t, :] 
        
        # Dự đoán tương lai: z_{t+1} = Predictor(z_t, a_t)
        z_next = model["predictor"](torch.cat([z_states, act_t], dim=1))
        
        # Tính điểm: Càng gần Goal càng tốt (Negative L2 Distance)
        # Vì LeJEPA học không gian Euclidean, ta dùng L2 trực tiếp
        dist = torch.norm(z_next - z_goal, dim=1)
        reward = -dist 
        
        # Cộng dồn điểm (có discount factor nhẹ nếu muốn, ở đây = 1)
        cumulative_rewards += reward
        z_states = z_next

    # 3. Chọn hành động (Weighted Average)
    # Softmax trên reward để lấy trọng số
    scores = F.softmax(cumulative_rewards / TEMPERATURE, dim=0)
    
    # Nhân trọng số vào hành động đầu tiên
    # actions[:, 0, :] shape [N, 4]
    # scores shape [N] -> unsqueeze thành [N, 1]
    best_action = torch.sum(actions[:, 0, :] * scores.unsqueeze(1), dim=0)
    
    return best_action.detach().cpu().numpy()

# --- MAIN LOOP ---
def run_inference():
    # Load Model
    print("🧠 Loading Brain...")
    base = LeJEPA_Robot(action_dim=4, z_dim=64).to(DEVICE)
    base.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    base.eval()

    # Build callables (maybe DataParallel)
    encoder = base.encoder
    predictor = base.predictor
    if USE_DATAPARALLEL and DEVICE == "cuda" and torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        predictor = torch.nn.DataParallel(predictor)
    model = {"encoder": encoder, "predictor": predictor}
    
    env, ml1 = get_env()
    model_mujoco = env.unwrapped.model 
    data = env.unwrapped.data
    renderer = mujoco.Renderer(model_mujoco, height=RENDER_SIZE[0], width=RENDER_SIZE[1])
    
    if PREVIEW:
        cv2.namedWindow("mpc_preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("mpc_preview", IMG_SIZE[0] * PREVIEW_SCALE * 2 + 10, IMG_SIZE[1] * PREVIEW_SCALE)

    # Loop theo episode giống 01_collect_data.py
    for ep in range(NUM_EPISODES):
        task = random.choice(ml1.train_tasks)
        env.set_task(task)
        env.reset()

        print(f"📸 [ep={ep}] Capturing Goal Image...")
        img_goal, img_goal_chw = capture_goal_image(env, renderer, CAMERA_NAME)
        with torch.no_grad():
            z_goal = model["encoder"](img_goal)

        if PREVIEW:
            goal_vis = chw_rgb_to_hwc_bgr(img_goal_chw)
            if PREVIEW_SCALE and PREVIEW_SCALE != 1:
                goal_vis = cv2.resize(
                    goal_vis,
                    (IMG_SIZE[0] * PREVIEW_SCALE, IMG_SIZE[1] * PREVIEW_SCALE),
                    interpolation=cv2.INTER_NEAREST,
                )

        print(f"🚀 [ep={ep}] Start Control Loop...")
        for step in range(MAX_STEPS):
            img_curr_chw = grab_frame_np(renderer, data, CAMERA_NAME)
            img_curr = torch.tensor(img_curr_chw).float().to(DEVICE).unsqueeze(0) / 255.0

            with torch.no_grad():
                z_curr = model["encoder"](img_curr)
                action = plan_with_mppi(model, z_curr, z_goal)

            action = np.clip(action, -1.0, 1.0)

            step_ret = env.step(action)
            terminated = truncated = False
            if isinstance(step_ret, (tuple, list)) and len(step_ret) == 5:
                _, _, terminated, truncated, _ = step_ret
            elif isinstance(step_ret, (tuple, list)) and len(step_ret) == 4:
                _, _, done, _ = step_ret
                terminated = bool(done)
            else:
                raise RuntimeError(f"Unexpected env.step() return: {type(step_ret)} {step_ret}")

            if PREVIEW:
                curr_vis = chw_rgb_to_hwc_bgr(img_curr_chw)
                if PREVIEW_SCALE and PREVIEW_SCALE != 1:
                    curr_vis = cv2.resize(
                        curr_vis,
                        (IMG_SIZE[0] * PREVIEW_SCALE, IMG_SIZE[1] * PREVIEW_SCALE),
                        interpolation=cv2.INTER_NEAREST,
                    )
                sep = np.zeros((curr_vis.shape[0], 10, 3), dtype=np.uint8)
                vis = np.concatenate([curr_vis, sep, goal_vis], axis=1)
                cv2.putText(
                    vis,
                    f"ep={ep} step={step}  curr | goal",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("mpc_preview", vis)
                if (cv2.waitKey(PREVIEW_WAIT_MS) & 0xFF) in (ord("q"), 27):
                    return

            obs_dict = env.unwrapped._get_obs_dict()
            obj_pos = np.asarray(obs_dict["state_observation"], dtype=np.float32)[4:7]
            goal_pos = np.asarray(obs_dict["state_desired_goal"], dtype=np.float32)
            dist = float(np.linalg.norm(obj_pos - goal_pos))
            print(f"ep={ep} step={step}: Action {action[:2]}... | Dist to Goal: {dist:.4f}")

            if dist < 0.05:
                print("✅ SUCCESSSSS! AI đã đẩy trúng đích!")
                break

            if truncated or terminated:
                break

    if PREVIEW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()