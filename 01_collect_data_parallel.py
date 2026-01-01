import os
import time
import random
import multiprocessing as mp
import shutil

import numpy as np
import cv2
import mujoco
import metaworld
from metaworld.policies import SawyerPushV3Policy

from config import COMMON, COLLECT

from tqdm import tqdm


def grab_frame(renderer, data, cam_name, img_size):
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()  # HWC RGB
    img = np.flipud(img)     # fix upside-down
    img = cv2.resize(img, img_size)
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img.astype(np.uint8, copy=False)


def get_action_logic(env, policy, obs_vector, epsilon, noise_scale):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        obs_dict = env.unwrapped._get_obs_dict()
        obs_vector[-3:] = obs_dict["state_desired_goal"]
        base_action = policy.get_action(obs_vector)

        # add small exploration noise
        noise = np.random.normal(0.0, noise_scale, size=4)
        noise[3] = 0.0
        action = base_action + noise
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


def _open_worker_memmaps(shard_dir, worker_id, max_transitions, h, w):
    obs_path = os.path.join(shard_dir, f"obs_w{worker_id}.npy")
    actions_path = os.path.join(shard_dir, f"actions_w{worker_id}.npy")
    next_obs_path = os.path.join(shard_dir, f"next_obs_w{worker_id}.npy")

    obs_mm = np.lib.format.open_memmap(obs_path, mode="w+", dtype=np.uint8, shape=(max_transitions, 3, h, w))
    actions_mm = np.lib.format.open_memmap(actions_path, mode="w+", dtype=np.float32, shape=(max_transitions, 4))
    next_obs_mm = np.lib.format.open_memmap(next_obs_path, mode="w+", dtype=np.uint8, shape=(max_transitions, 3, h, w))
    return obs_mm, actions_mm, next_obs_mm, obs_path, actions_path, next_obs_path


def worker_collect(worker_id, num_episodes, seed, shard_dir):
    # Each worker must initialize its own env/renderer/policy.
    os.environ.setdefault("MUJOCO_GL", COMMON.mujoco_gl)

    random.seed(seed)
    np.random.seed(seed)

    env = None
    renderer = None
    obs_mm = actions_mm = next_obs_mm = None
    obs_path = actions_path = next_obs_path = None
    write_idx = 0
    try:
        ml1 = metaworld.ML1(COMMON.task_name)
        env = ml1.train_classes[COMMON.task_name](render_mode=None)
        model = env.unwrapped.model
        data = env.unwrapped.data
        renderer = mujoco.Renderer(model, height=COMMON.render_size[0], width=COMMON.render_size[1])
        policy = SawyerPushV3Policy()

        max_transitions = int(num_episodes) * int(COLLECT.max_steps)
        h, w = COMMON.img_size[1], COMMON.img_size[0]  # cv2 uses (W,H)
        obs_mm, actions_mm, next_obs_mm, obs_path, actions_path, next_obs_path = _open_worker_memmaps(
            shard_dir, worker_id, max_transitions, h, w
        )

        # Only print tqdm for worker_id == 0
        ep_range = range(num_episodes)
        iterator = tqdm(ep_range, desc=f"Worker {worker_id}", disable=(worker_id != 0 or num_episodes < 10))
        for _ep in iterator:
            try:
                task = random.choice(ml1.train_tasks)
                env.set_task(task)
                obs_vector, _ = env.reset()

                img_t = grab_frame(renderer, data, COMMON.camera_name, COMMON.img_size)
                for _ in range(COLLECT.max_steps):
                    if write_idx >= max_transitions:
                        break

                    action = get_action_logic(env, policy, obs_vector, COLLECT.epsilon, COLLECT.noise_scale)

                    step_ret = env.step(action)
                    # Gymnasium: obs, reward, terminated, truncated, info
                    if isinstance(step_ret, (tuple, list)) and len(step_ret) == 5:
                        obs_vector, _, terminated, truncated, _ = step_ret
                        if terminated or truncated:
                            break
                    else:
                        obs_vector = step_ret[0]

                    img_next = grab_frame(renderer, data, COMMON.camera_name, COMMON.img_size)

                    obs_mm[write_idx] = img_t
                    actions_mm[write_idx] = action
                    next_obs_mm[write_idx] = img_next
                    write_idx += 1

                    img_t = img_next
            except Exception:
                continue
    finally:
        # Ensure data is flushed to disk and resources are released even if the worker errors.
        try:
            if obs_mm is not None:
                obs_mm.flush()
            if actions_mm is not None:
                actions_mm.flush()
            if next_obs_mm is not None:
                next_obs_mm.flush()
        except Exception:
            pass
        try:
            if renderer is not None:
                renderer.close()
        except Exception:
            pass
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

    meta_path = os.path.join(shard_dir, f"meta_w{worker_id}.npz")
    np.savez_compressed(meta_path, n=np.array([write_idx], dtype=np.int64))
    return {"worker_id": worker_id, "n": write_idx, "obs": obs_path, "actions": actions_path, "next_obs": next_obs_path, "meta": meta_path}


def merge_shards(shard_results, out_dir):
    # Merge worker memmaps into one contiguous set of memmaps to match training loader.
    total = int(sum(r["n"] for r in shard_results))
    h, w = COMMON.img_size[1], COMMON.img_size[0]

    obs_out = os.path.join(out_dir, "obs.npy")
    actions_out = os.path.join(out_dir, "actions.npy")
    next_obs_out = os.path.join(out_dir, "next_obs.npy")
    meta_out = os.path.join(out_dir, "meta.npz")

    obs_mm = np.lib.format.open_memmap(obs_out, mode="w+", dtype=np.uint8, shape=(total, 3, h, w))
    actions_mm = np.lib.format.open_memmap(actions_out, mode="w+", dtype=np.float32, shape=(total, 4))
    next_obs_mm = np.lib.format.open_memmap(next_obs_out, mode="w+", dtype=np.uint8, shape=(total, 3, h, w))

    write = 0
    try:
        for r in sorted(shard_results, key=lambda x: x["worker_id"]):
            n = int(r["n"])
            if n <= 0:
                continue
            o = np.load(r["obs"], mmap_mode="r")[:n]
            a = np.load(r["actions"], mmap_mode="r")[:n]
            no = np.load(r["next_obs"], mmap_mode="r")[:n]

            obs_mm[write : write + n] = o
            actions_mm[write : write + n] = a
            next_obs_mm[write : write + n] = no
            write += n
    finally:
        try:
            obs_mm.flush()
            actions_mm.flush()
            next_obs_mm.flush()
        except Exception:
            pass

    np.savez_compressed(meta_out, n=np.array([write], dtype=np.int64))
    return write


def main():
    os.environ.setdefault("MUJOCO_GL", COMMON.mujoco_gl)

    out_dir = COMMON.data_dir
    os.makedirs(out_dir, exist_ok=True)
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    total_episodes = int(COLLECT.num_episodes)
    num_workers = int(COLLECT.num_workers) if COLLECT.num_workers else int(mp.cpu_count())
    num_workers = max(1, num_workers)
    episodes_per_worker = total_episodes // num_workers
    remainder = total_episodes % num_workers

    args = []
    base_seed = 12345
    for wid in range(num_workers):
        n_ep = episodes_per_worker + (1 if wid < remainder else 0)
        if n_ep <= 0:
            continue
        args.append((wid, n_ep, base_seed + wid * 100000, shard_dir))

    print(f"Number of workers: {len(args)} | Total episodes: {total_episodes} | IMG_SIZE: {COMMON.img_size}")
    t0 = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(args)) as pool:
        try:
            results = pool.starmap(worker_collect, args)
        except KeyboardInterrupt:
            # Ensure we don't leave workers/semaphores behind on Ctrl+C.
            pool.terminate()
            pool.join()
            raise

    n_total = merge_shards(results, out_dir)
    if getattr(COLLECT, "cleanup_shards", True):
        # shards are only an intermediate format; remove to save disk/inodes
        try:
            shutil.rmtree(shard_dir, ignore_errors=True)
        except Exception:
            pass
    dt = time.time() - t0
    print(f"Done. transitions={n_total} time={dt:.1f}s transitions/s={n_total/dt:.1f}")


if __name__ == "__main__":
    main()

