import argparse
import os
import numpy as np
import cv2
from config import COMMON, VIEW


def to_hwc_uint8(img_chw: np.ndarray) -> np.ndarray:
    """
    Convert CHW uint8 RGB -> HWC uint8 BGR for cv2.imshow.
    """
    if img_chw.ndim != 3 or img_chw.shape[0] != 3:
        raise ValueError(f"Expected CHW with C=3, got shape={img_chw.shape}")
    if img_chw.dtype != np.uint8:
        img_chw = img_chw.astype(np.uint8, copy=False)
    img_hwc_rgb = np.transpose(img_chw, (1, 2, 0))
    return cv2.cvtColor(img_hwc_rgb, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="View obs/next_obs images from a .npz buffer.")
    ap.add_argument(
        "--npz",
        default=os.path.join(COMMON.data_dir, "train_buffer.npz"),
        help="Path to .npz (default: <data_dir>/train_buffer.npz)",
    )
    ap.add_argument(
        "--npy_dir",
        default=None,
        help="Directory containing obs.npy/next_obs.npy/actions.npy/meta.npz produced by 01_collect_data.py (STREAM_TO_NPY).",
    )
    ap.add_argument("--idx", type=int, default=0, help="Start index (default: 0)")
    ap.add_argument("--stride", type=int, default=VIEW.default_stride, help="Step size when pressing n/p")
    ap.add_argument("--scale", type=int, default=VIEW.default_scale, help="Resize factor for display")
    args = ap.parse_args()

    if args.npy_dir:
        d = args.npy_dir
        obs_path = os.path.join(d, "obs.npy")
        next_obs_path = os.path.join(d, "next_obs.npy")
        meta_path = os.path.join(d, "meta.npz")

        if not (os.path.exists(obs_path) and os.path.exists(next_obs_path) and os.path.exists(meta_path)):
            raise SystemExit(f"Missing files in {d}. Expected obs.npy, next_obs.npy, meta.npz")

        meta = np.load(meta_path)
        n = int(meta["n"][0])
        obs = np.load(obs_path, mmap_mode="r")[:n]
        next_obs = np.load(next_obs_path, mmap_mode="r")[:n]
    else:
        if not os.path.exists(args.npz):
            raise SystemExit(f"File not found: {args.npz}")

        buf = np.load(args.npz)
        if "obs" not in buf or "next_obs" not in buf:
            raise SystemExit(f"Expected keys 'obs' and 'next_obs' in {args.npz}. Found: {list(buf.keys())}")

        obs = buf["obs"]
        next_obs = buf["next_obs"]

    if obs.shape != next_obs.shape:
        raise SystemExit(f"obs.shape != next_obs.shape: {obs.shape} vs {next_obs.shape}")
    if obs.ndim != 4 or obs.shape[1] != 3:
        raise SystemExit(f"Expected shape (N, 3, H, W). Got: {obs.shape}")

    n = obs.shape[0]
    idx = max(0, min(args.idx, n - 1))

    win = "buffer_view (n: next, p: prev, q/esc: quit)"
    while True:
        o = to_hwc_uint8(obs[idx])
        no = to_hwc_uint8(next_obs[idx])

        if args.scale and args.scale != 1:
            o = cv2.resize(o, (o.shape[1] * args.scale, o.shape[0] * args.scale), interpolation=cv2.INTER_NEAREST)
            no = cv2.resize(no, (no.shape[1] * args.scale, no.shape[0] * args.scale), interpolation=cv2.INTER_NEAREST)

        sep = np.zeros((o.shape[0], 10, 3), dtype=np.uint8)
        vis = np.concatenate([o, sep, no], axis=1)

        cv2.putText(
            vis,
            f"idx={idx}/{n-1}  obs | next_obs",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win, vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("n"):
            idx = min(n - 1, idx + max(1, args.stride))
        elif key == ord("p"):
            idx = max(0, idx - max(1, args.stride))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


