"""
Shared helpers used by scripts 2-7.
Import with `import common` (scripts are run from the project root as `python scripts/N_name.py`).
"""

import os
import json
import hashlib
import pickle
from pathlib import Path

import numpy as np
import yaml

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pick_device(requested="auto"):
    """Resolve 'auto' to the best available torch backend: xpu (Intel Arc) > cuda (NVIDIA) > cpu."""
    import torch
    if requested and requested != "auto":
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)
    )


def mvtec_paths(cfg):
    """Return the key folders for the configured category."""
    cat = cfg["dataset"]["name"]
    root = os.path.join(cfg["paths"]["mvtec_dir"], cat)
    return {
        "category": cat,
        "root": root,
        "train_good": os.path.join(root, "train", "good"),
        "test": os.path.join(root, "test"),
        "test_good": os.path.join(root, "test", "good"),
        "ground_truth": os.path.join(root, "ground_truth"),
    }


def defect_classes(cfg):
    """Defect type folder names under test/, excluding 'good'."""
    p = mvtec_paths(cfg)
    if not os.path.isdir(p["test"]):
        return []
    return sorted(
        d for d in os.listdir(p["test"])
        if d != "good" and os.path.isdir(os.path.join(p["test"], d))
    )


def mask_path_for(image_path, cfg):
    """MVTec convention: test/<cls>/000.png  ->  ground_truth/<cls>/000_mask.png"""
    p = mvtec_paths(cfg)
    cls = os.path.basename(os.path.dirname(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(p["ground_truth"], cls, f"{stem}_mask.png")


def apply_seed(cfg, seed):
    """CLI --seed overrides config; every seed gets its own split, synthetic folders and result files."""
    if seed is not None:
        cfg["dataset"]["seed"] = int(seed)
    return cfg


def seed_tag(cfg):
    return f"s{cfg['dataset']['seed']}"


def synthetic_root(cfg, refined=False):
    base = cfg["paths"]["synthetic_refined_dir" if refined else "synthetic_dir"]
    return os.path.join(base, f"{cfg['dataset']['name']}_{seed_tag(cfg)}")


def split_file(cfg):
    return os.path.join(cfg["paths"]["splits_dir"],
                        f"{cfg['dataset']['name']}_k{cfg['dataset']['k_shot']}_{seed_tag(cfg)}.json")


def load_split(cfg):
    path = split_file(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split file not found: {path}\nRun: python scripts/2_make_splits_and_synthetic.py --seed {cfg['dataset']['seed']}"
        )
    with open(path) as f:
        return json.load(f)


def ensure_dirs(cfg):
    for key in ("synthetic_dir", "synthetic_refined_dir", "splits_dir", "cache_dir", "models_dir", "results_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Feature extraction (torch). Kept in one place so every script uses identical features.
# --------------------------------------------------------------------------------------

class FeatureExtractor:
    """
    ResNet50 (ImageNet weights).
    - global_features(paths): (N, 2048) pooled vector per image, cached on disk by file hash.
    - patch_features(paths):  (N, P, C) mid-level patch descriptors for the anomaly stage.
    """

    def __init__(self, cfg, backbone=None, input_size=None):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        self.cfg = cfg
        self.device = torch.device(pick_device(cfg["features"].get("device", "auto")))
        size = int(input_size or cfg["features"]["input_size"])
        self.backbone = backbone or cfg["features"].get("backbone", "resnet50")
        self.size = size

        if self.backbone == "wide_resnet50_2":
            net = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        net = net.eval().to(self.device)
        self.net = net
        # Explicit stages so we can tap intermediate layers without hooks.
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layers = {"layer1": net.layer1, "layer2": net.layer2, "layer3": net.layer3, "layer4": net.layer4}

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.cache_dir = cfg["paths"]["cache_dir"]
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        print(f"[features] device={self.device}, backbone={self.backbone}, input={size}px")

    # ---- helpers ----
    def _load(self, path):
        from PIL import Image
        return self.transform(Image.open(path).convert("RGB"))

    def _cache_key(self, path, tag):
        st = os.stat(path)
        h = hashlib.md5(f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{tag}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    # ---- global pooled features (stage 2 classifier) ----
    def global_features(self, paths, batch_size=16, desc="features", use_cache=True):
        import torch
        from tqdm import tqdm

        out = np.zeros((len(paths), 2048), dtype=np.float32)
        todo = []
        for i, p in enumerate(paths):
            ck = self._cache_key(p, f"global_{self.backbone}_{self.size}")
            if use_cache and os.path.exists(ck):
                out[i] = np.load(ck)
            else:
                todo.append(i)

        for s in tqdm(range(0, len(todo), batch_size), desc=desc, disable=len(todo) == 0):
            idx = todo[s:s + batch_size]
            x = torch.stack([self._load(paths[i]) for i in idx]).to(self.device)
            with torch.no_grad():
                f = self.stem(x)
                for name in ("layer1", "layer2", "layer3", "layer4"):
                    f = self.layers[name](f)
                f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
            f = f.float().cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = f[j]
                if use_cache:
                    np.save(self._cache_key(paths[i], f"global_{self.backbone}_{self.size}"), f[j])
        return out

    # ---- patch features (stage 1 anomaly detector) ----
    def patch_features(self, paths, layer_names=("layer2", "layer3"), batch_size=8, desc="patches"):
        """
        PatchCore-style locally-aware patch descriptors:
        each tapped layer is 3x3 average-pooled, upsampled to the largest map, concatenated.
        Returns float32 array (N, H*W, C).
        """
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        feats = []
        for s in tqdm(range(0, len(paths), batch_size), desc=desc):
            x = torch.stack([self._load(p) for p in paths[s:s + batch_size]]).to(self.device)
            with torch.no_grad():
                f = self.stem(x)
                taps = []
                for name in ("layer1", "layer2", "layer3", "layer4"):
                    f = self.layers[name](f)
                    if name in layer_names:
                        taps.append(F.avg_pool2d(f, kernel_size=3, stride=1, padding=1))
                    if name == layer_names[-1]:
                        break
                H, W = taps[0].shape[-2:]
                taps = [t if t.shape[-2:] == (H, W) else F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
                        for t in taps]
                f = torch.cat(taps, dim=1)                # (B, C, H, W)
                f = f.flatten(2).transpose(1, 2)          # (B, H*W, C)
            feats.append(f.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


def is_valid_image(path):
    """True if the file exists and PIL can fully decode it (catches truncated files from interrupted runs)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def atomic_save(pil_image, path):
    """Write to a temp file in the same folder, then rename: a crash mid-write never leaves a half PNG behind."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp.png"
    pil_image.save(tmp, format="PNG")
    os.replace(tmp, path)


def save_pickle(obj, path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------------------
# Stage-1 memory bank persistence and patch-level localisation (used by scripts 4 and 5)
# --------------------------------------------------------------------------------------

def bank_path(cfg):
    return os.path.join(cfg["paths"]["models_dir"], "stage1_bank.npz")


def save_bank(cfg, bank, grid_hw, meta):
    Path(cfg["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    np.savez(bank_path(cfg), bank=bank.astype(np.float32), grid_hw=np.array(grid_hw), meta=json.dumps(meta))


def load_bank(cfg):
    p = bank_path(cfg)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Stage-1 memory bank not found at {p}. Run: python scripts/4_anomaly_stage.py")
    d = np.load(p, allow_pickle=False)
    return d["bank"], tuple(int(x) for x in d["grid_hw"]), json.loads(str(d["meta"]))


def patch_min_distances(fx, paths, bank, layer_names, desc="patch distances"):
    """
    For each image: per-patch distance to the nearest memory-bank patch.
    Returns (N, P) float32 and the (H, W) patch grid.
    """
    import torch
    device = fx.device
    bank_t = torch.from_numpy(bank).to(device)
    bank_sq = (bank_t ** 2).sum(1)
    feats = fx.patch_features(paths, layer_names=tuple(layer_names), desc=desc)   # (N, P, C)
    N, P, C = feats.shape
    out = np.zeros((N, P), dtype=np.float32)
    for i in range(N):
        f_t = torch.from_numpy(feats[i]).to(device)
        f_sq = (f_t ** 2).sum(1, keepdim=True)
        d_min = torch.full((P,), float("inf"), device=device)
        for s_ in range(0, bank_t.shape[0], 8192):
            b = bank_t[s_:s_ + 8192]
            d = f_sq + bank_sq[s_:s_ + 8192][None, :] - 2.0 * f_t @ b.T
            d_min = torch.minimum(d_min, d.min(1).values)
        out[i] = d_min.clamp_min(0).sqrt().cpu().numpy()
    g = int(round(P ** 0.5))
    return out, (g, g)


def localize_from_distances(dist_row, grid_hw):
    """Fraction (x, y) in [0,1] of the most anomalous patch centre."""
    H, W = grid_hw
    idx = int(np.argmax(dist_row))
    r, c = divmod(idx, W)
    return (c + 0.5) / W, (r + 0.5) / H


def mask_centre_fraction(mask_path):
    """Fraction (x, y) of the centroid of a binary mask file, or None."""
    import cv2
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None or (m > 0).sum() == 0:
        return None
    ys, xs = np.nonzero(m > 0)
    return float(xs.mean()) / m.shape[1], float(ys.mean()) / m.shape[0]


def crop_around(image_path, centre_frac, crop_frac, out_size, out_path):
    """Square crop of side crop_frac*min(H,W) centred on centre_frac (clamped inside the image), resized to out_size."""
    from PIL import Image
    if os.path.exists(out_path):
        if is_valid_image(out_path):
            return out_path
        os.remove(out_path)
    img = Image.open(image_path).convert("RGB")
    Wd, Ht = img.size
    side = int(round(crop_frac * min(Wd, Ht)))
    cx, cy = centre_frac[0] * Wd, centre_frac[1] * Ht
    x0 = int(round(min(max(cx - side / 2, 0), Wd - side)))
    y0 = int(round(min(max(cy - side / 2, 0), Ht - side)))
    atomic_save(img.crop((x0, y0, x0 + side, y0 + side)).resize((out_size, out_size), Image.LANCZOS), out_path)
    return out_path