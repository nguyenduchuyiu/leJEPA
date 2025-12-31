import os
# MuJoCo default GLX often fails with errors like:
# "GLX: No GLXFBConfigs returned" / "gladLoadGL error".
# EGL works both headless and with a desktop display, so use it by default
# unless the user explicitly set MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import metaworld
import random
import cv2
import mujoco 
from metaworld.policies import SawyerPushV3Policy
from tqdm import tqdm
from config import COMMON, COLLECT

# --- CONFIG ---
TASK_NAME = COMMON.task_name
NUM_EPISODES = COLLECT.num_episodes
MAX_STEPS = COLLECT.max_steps
IMG_SIZE = COMMON.img_size
RENDER_SIZE = COMMON.render_size
SAVE_DIR = COMMON.data_dir
CAMERA_NAME = COMMON.camera_name
NOISE_SCALE = COLLECT.noise_scale
EPSILON = COLLECT.epsilon

# For large runs (e.g. NUM_EPISODES=5000), keeping all frames in RAM and then
# calling np.savez_compressed will often trigger OOM and get the process killed.
# Stream-to-disk avoids that by writing into .npy memmaps during collection.
STREAM_TO_NPY = COLLECT.stream_to_npy

# Visual debugging:
# - If you have a display, set PREVIEW = True to see a live cv2 window.
# - On headless machines, keep PREVIEW = False.
PREVIEW = COLLECT.preview
PREVIEW_WAIT_MS = COLLECT.preview_wait_ms
PREVIEW_SCALE = COLLECT.preview_scale  # phóng to cửa sổ preview (1 = giữ nguyên)

def get_env():
    ml1 = metaworld.ML1(TASK_NAME)
    # Render mode chỉ để cho có thủ tục, ta sẽ render thủ công
    env = ml1.train_classes[TASK_NAME](render_mode=None)
    
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    return env

# --- HÀM MỚI QUAN TRỌNG NHẤT ---
def grab_frame(renderer, data, cam_name):
    """
    Dùng mujoco.Renderer trực tiếp thay vì env.render()
    """
    # 1. Cập nhật scene với dữ liệu vật lý hiện tại (data) và camera mong muốn
    renderer.update_scene(data, camera=cam_name)
    
    # 2. Render ra ảnh
    img = renderer.render() # Trả về (H, W, 3) RGB

    # Backend trả ảnh bị lật dọc
    img = np.flipud(img)
    
    # 3. Resize về 64x64
    img = cv2.resize(img, IMG_SIZE)
    
    # 4. Chuyển sang Channel-First (3, 64, 64) cho PyTorch
    img = np.transpose(img, (2, 0, 1)) 
    return img

def chw_rgb_to_hwc_bgr(img_chw: np.ndarray) -> np.ndarray:
    """Convert CHW RGB uint8 -> HWC BGR uint8 for OpenCV display/video."""
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    return cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

# --- SỬA LẠI HÀM COLLECT ---
def collect():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Khởi tạo Env
    ml1 = metaworld.ML1(TASK_NAME)
    env = ml1.train_classes[TASK_NAME](render_mode=None)

    # Khởi tạo các thành phần render/policy như cũ
    model = env.unwrapped.model
    data = env.unwrapped.data
    renderer = mujoco.Renderer(model, height=RENDER_SIZE[0], width=RENDER_SIZE[1])

    policy = SawyerPushV3Policy()

    max_transitions = int(NUM_EPISODES) * int(MAX_STEPS)
    write_idx = 0

    obs_mm = actions_mm = next_obs_mm = None
    data_obs = data_actions = data_next_obs = None

    if STREAM_TO_NPY:
        # Note: .npy is written as a memory-mapped array on disk (low RAM).
        obs_path = os.path.join(SAVE_DIR, "obs.npy")
        actions_path = os.path.join(SAVE_DIR, "actions.npy")
        next_obs_path = os.path.join(SAVE_DIR, "next_obs.npy")

        # grab_frame returns CHW (3, H, W). IMG_SIZE is (W, H) in cv2.resize().
        h, w = IMG_SIZE[1], IMG_SIZE[0]
        obs_mm = np.lib.format.open_memmap(obs_path, mode="w+", dtype=np.uint8, shape=(max_transitions, 3, h, w))
        actions_mm = np.lib.format.open_memmap(actions_path, mode="w+", dtype=np.float32, shape=(max_transitions, 4))
        next_obs_mm = np.lib.format.open_memmap(next_obs_path, mode="w+", dtype=np.uint8, shape=(max_transitions, 3, h, w))
    else:
        data_obs, data_actions, data_next_obs = [], [], []

    print(f"🚀 Bắt đầu thu thập dữ liệu (Đã fix lỗi lặp task)...")

    if PREVIEW:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("preview", IMG_SIZE[0] * PREVIEW_SCALE, IMG_SIZE[1] * PREVIEW_SCALE)

    try:
        for ep in tqdm(range(NUM_EPISODES)):
            try:
                # --- FIX 1: RANDOM TASK MỖI TẬP ---
                # Phải set task mới thì vị trí vật thể/goal mới thay đổi
                task = random.choice(ml1.train_tasks)
                env.set_task(task)

                # Reset môi trường
                obs_vector, _ = env.reset()

                # Render frame đầu
                img_t = grab_frame(renderer, data, CAMERA_NAME)
                if PREVIEW:
                    vis = chw_rgb_to_hwc_bgr(img_t)
                    if PREVIEW_SCALE and PREVIEW_SCALE != 1:
                        vis = cv2.resize(vis, (IMG_SIZE[0] * PREVIEW_SCALE, IMG_SIZE[1] * PREVIEW_SCALE), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("preview", vis)
                    if (cv2.waitKey(PREVIEW_WAIT_MS) & 0xFF) in (ord("q"), 27):
                        raise KeyboardInterrupt

                for _ in range(MAX_STEPS):
                    if write_idx >= max_transitions:
                        break

                    # --- LOGIC MIX ACTION (Giữ nguyên như đã bàn) ---
                    if np.random.rand() < EPSILON:
                        action = env.action_space.sample()
                    else:
                        # Đây là Ground Truth (Sự thật tuyệt đối)
                        obs_dict = env.unwrapped._get_obs_dict()
                        real_goal = obs_dict['state_desired_goal']

                        # --- BƯỚC 2: PHẪU THUẬT VECTOR ---
                        # Ghi đè 3 phần tử cuối cùng của vector bằng Goal thật
                        # Policy V3 sẽ đọc 3 số này để định hướng
                        obs_vector[-3:] = real_goal
                        base_action = policy.get_action(obs_vector)
                        xyz_noise = np.random.normal(0, NOISE_SCALE, size=3)
                        gripper_noise = 0.0 # Khóa gripper
                        noise = np.hstack([xyz_noise, gripper_noise])
                        action = base_action + noise

                    action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)

                    # Step
                    obs_vector, _, _, _, _ = env.step(action)
                    img_next = grab_frame(renderer, data, CAMERA_NAME)
                    if PREVIEW:
                        vis = chw_rgb_to_hwc_bgr(img_next)
                        if PREVIEW_SCALE and PREVIEW_SCALE != 1:
                            vis = cv2.resize(vis, (IMG_SIZE[0] * PREVIEW_SCALE, IMG_SIZE[1] * PREVIEW_SCALE), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("preview", vis)
                        if (cv2.waitKey(PREVIEW_WAIT_MS) & 0xFF) in (ord("q"), 27):
                            raise KeyboardInterrupt

                    # Lưu
                    if STREAM_TO_NPY:
                        obs_mm[write_idx] = img_t
                        actions_mm[write_idx] = action
                        next_obs_mm[write_idx] = img_next
                    else:
                        data_obs.append(img_t)
                        data_actions.append(action)
                        data_next_obs.append(img_next)

                    write_idx += 1
                    img_t = img_next

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"⚠️ Lỗi episode {ep}: {e}")
                continue
    finally:
        # Avoid EGL errors on interpreter shutdown
        try:
            renderer.close()
        except Exception:
            pass
        if PREVIEW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    if STREAM_TO_NPY:
        print("💾 Đang flush và lưu metadata...")
        obs_mm.flush()
        actions_mm.flush()
        next_obs_mm.flush()
        meta_path = os.path.join(SAVE_DIR, "meta.npz")
        np.savez(meta_path, n=np.array([write_idx], dtype=np.int64))
        print(f"✅ Xong! Lưu dạng memmap tại: {SAVE_DIR} (n={write_idx})")
    else:
        print("📦 Đang nén và lưu dữ liệu...")
        np_obs = np.array(data_obs, dtype=np.uint8)
        np_actions = np.array(data_actions, dtype=np.float32)
        np_next_obs = np.array(data_next_obs, dtype=np.uint8)

        save_path = os.path.join(SAVE_DIR, 'train_buffer.npz')
        np.savez_compressed(
            save_path,
            obs=np_obs,
            actions=np_actions,
            next_obs=np_next_obs
        )

        print(f"✅ Xong! File lưu tại: {save_path}")
        print(f"📊 Shape ảnh: {np_obs.shape}")

if __name__ == "__main__":
    collect()