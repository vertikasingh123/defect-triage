"""
Script 2: Build the few-shot split and generate label-preserving synthetic defects.

  python scripts/2_make_splits_and_synthetic.py [--seed N]

For each defect class in test/<class>/:
  - the first k_shot images (deterministic, seeded shuffle) become the REAL training examples
  - every other image is HELD OUT for testing (never touched again until evaluation)
  - synthetic images are made by cutting the real defect out of each training example
    (using MVTec's ground_truth mask) and pasting it onto random clean images from train/good,
    ALIGNED to the destination part's position and orientation (so a head defect lands on the head
    even when the part is rotated), with mild jitter and a feathered edge. Pastes that would miss the
    part are rejected. The paste mask is saved next to each image (in <class>/_masks/) so script 3
    can restrict diffusion to that region.

Output:
  data/splits/<category>_k<k>.json
  data/synthetic/<category>/<class>/*.png
"""

import os
import sys
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def make_split(cfg):
    p = common.mvtec_paths(cfg)
    classes = common.defect_classes(cfg)
    if not classes:
        sys.exit(f"No defect classes found under {p['test']}. Run script 1 first.")

    k = cfg["dataset"]["k_shot"]
    rng = random.Random(cfg["dataset"]["seed"])

    split = {"category": p["category"], "k_shot": k, "classes": classes, "train": {}, "test": {}}
    print(f"Category: {p['category']}   k-shot: {k}")
    print(f"{'class':<20}{'total':>7}{'train':>7}{'test':>7}")
    for cls in classes:
        imgs = common.list_images(os.path.join(p["test"], cls))
        # keep only images that have a ground-truth mask (needed for cut-paste)
        imgs = [i for i in imgs if os.path.exists(common.mask_path_for(i, cfg))]
        rng.shuffle(imgs)
        if len(imgs) <= k:
            sys.exit(f"Class '{cls}' has only {len(imgs)} images; k_shot={k} leaves nothing to test on.")
        split["train"][cls] = sorted(imgs[:k])
        split["test"][cls] = sorted(imgs[k:])
        print(f"{cls:<20}{len(imgs):>7}{k:>7}{len(imgs) - k:>7}")

    split["test_good"] = common.list_images(p["test_good"])
    out = common.split_file(cfg)
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nSplit saved -> {out}")
    return split


# ----------------------------------------------------------------------------------------------
# Part pose (so the defect lands ON the part, at the matching place, even if the part is rotated)
# ----------------------------------------------------------------------------------------------

def foreground_mask(img_bgr):
    """Binary mask of the part. MVTec objects sit on a fairly uniform background; Otsu + largest blob."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # the part is whichever class touches the image border LESS (background touches the border)
    border = np.concatenate([th[0], th[-1], th[:, 0], th[:, -1]])
    if border.mean() > 127:
        th = 255 - th
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        th = (lab == i).astype(np.uint8) * 255
    return th


def part_pose(mask):
    """
    Return (centroid[2], angle_rad, length_scale) from the mask's principal axis.
    The axis direction is disambiguated toward the WIDER end (e.g. a screw head), so
    'head' maps to 'head' between two parts.
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, 1]
    perp = evecs[:, 0]
    proj = (pts - mean) @ axis
    off = np.abs((pts - mean) @ perp)
    width_pos = off[proj > 0].mean() if (proj > 0).any() else 0
    width_neg = off[proj < 0].mean() if (proj < 0).any() else 0
    if width_neg > width_pos:
        axis = -axis
    angle = float(np.arctan2(axis[1], axis[0]))
    scale = float(np.sqrt(max(evals[1], 1e-6)))          # along the part (length)
    width = float(3.5 * np.sqrt(max(evals[0], 1e-6)))    # across the part (approx. full width in px)
    return mean, angle, scale, width


def align_affine(src_pose, dst_pose, extra_rot_deg=0.0, extra_scale=1.0, shift=(0, 0), scale_clip=(0.8, 1.25)):
    """2x3 affine mapping source-part coordinates onto the destination part (+ small jitter)."""
    (ms, as_, ss, _), (md, ad, sd, _) = src_pose, dst_pose
    theta = (ad - as_) + np.deg2rad(extra_rot_deg)
    s = float(np.clip(sd / ss, *scale_clip)) * extra_scale
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) * s
    t = md + np.asarray(shift, dtype=np.float64) - R @ ms
    return np.hstack([R, t[:, None]]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Cut-paste
# ----------------------------------------------------------------------------------------------

def cut_paste(src_img, src_mask, dst_img, cfg, rng, min_overlap=None):
    """
    Paste the masked defect of src_img onto dst_img at the corresponding place on the destination part.
    Returns (image, paste_mask) or None if the defect would not land on the part.
    """
    S = cfg["synthetic"]["output_size"]
    src = cv2.resize(src_img, (S, S), interpolation=cv2.INTER_AREA)
    dst = cv2.resize(dst_img, (S, S), interpolation=cv2.INTER_AREA)
    m = cv2.resize(src_mask, (S, S), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8) * 255
    if m.sum() == 0:
        return None

    src_fg, dst_fg = foreground_mask(src), foreground_mask(dst)
    if src_fg.sum() < 500 or dst_fg.sum() < 500:
        return None
    src_pose, dst_pose = part_pose(src_fg), part_pose(dst_fg)

    # 1) exact alignment: source part frame -> destination part frame
    M = align_affine(src_pose, dst_pose)
    src_w = cv2.warpAffine(src, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m_w = cv2.warpAffine(m, M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if m_w.sum() == 0:
        return None

    # 2) small jitter ABOUT THE DEFECT, scaled to the part's width (so a thin part gets a small shift)
    ys, xs = np.nonzero(m_w)
    cx, cy = float(xs.mean()), float(ys.mean())
    part_w = dst_pose[3]
    shift = cfg["synthetic"]["paste_jitter"] * part_w
    Mj = cv2.getRotationMatrix2D((cx, cy),
                                 rng.uniform(-cfg["synthetic"]["rotate_deg"], cfg["synthetic"]["rotate_deg"]),
                                 rng.uniform(*cfg["synthetic"]["scale_range"]))
    Mj[:, 2] += (rng.uniform(-shift, shift), rng.uniform(-shift, shift))
    src_w = cv2.warpAffine(src_w, Mj, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m_w = cv2.warpAffine(m_w, Mj, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if m_w.sum() == 0:
        return None

    # 3) reject pastes that do not land on the destination part (no tolerance: shadows are not the part)
    if min_overlap is None:
        min_overlap = cfg["synthetic"].get("min_on_part", 0.9)
    overlap = (m_w > 0) & (dst_fg > 0)
    if overlap.sum() / max((m_w > 0).sum(), 1) < min_overlap:
        return None

    # feather scaled to defect size so thin scratches are not blurred away
    area = float((m_w > 0).sum())
    fp = int(np.clip(np.sqrt(area) / 4, 3, cfg["synthetic"]["feather_px"])) | 1
    alpha = cv2.GaussianBlur(m_w.astype(np.float32) / 255.0, (fp, fp), 0)[..., None]

    # lighting match: scalar luminance gain measured on the RING around the paste (surface vs surface),
    # so the two photos' lighting agrees without recolouring the defect itself
    ring = (cv2.dilate(m_w, np.ones((15, 15), np.uint8)) > 0) & (m_w == 0) & (dst_fg > 0)
    if ring.sum() > 50:
        g_dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)[ring].mean() + 1e-6
        g_src = cv2.cvtColor(src_w, cv2.COLOR_BGR2GRAY)[ring].mean() + 1e-6
        gain = float(np.clip(g_dst / g_src, 0.75, 1.3))
        src_w = np.clip(src_w.astype(np.float32) * gain, 0, 255)

    out = alpha * src_w.astype(np.float32) + (1 - alpha) * dst.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), m_w


def generate_synthetic(cfg, split):
    p = common.mvtec_paths(cfg)
    rng = random.Random(cfg["dataset"]["seed"] + 1)
    good = common.list_images(p["train_good"])
    if not good:
        sys.exit(f"No clean images in {p['train_good']}")

    out_root = common.synthetic_root(cfg)
    per_class = cfg["synthetic"]["per_class"]
    total_written = 0

    for cls in split["classes"]:
        out_dir = os.path.join(out_root, cls)
        mask_dir = os.path.join(out_dir, "_masks")
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
        # clear stale files so re-runs are reproducible
        for d in (out_dir, mask_dir):
            for f in os.listdir(d):
                fp_ = os.path.join(d, f)
                if os.path.isfile(fp_):
                    os.remove(fp_)

        sources = split["train"][cls]
        loaded = []
        for sp in sources:
            img = cv2.imread(sp, cv2.IMREAD_COLOR)
            msk = cv2.imread(common.mask_path_for(sp, cfg), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                print(f"  ! skip unreadable {sp}")
                continue
            loaded.append((os.path.splitext(os.path.basename(sp))[0], img, msk))

        written = 0
        attempts = 0
        pbar = tqdm(total=per_class, desc=f"synthetic/{cls}")
        while written < per_class and attempts < per_class * 12:
            attempts += 1
            stem, img, msk = loaded[written % len(loaded)] if attempts <= per_class else rng.choice(loaded)
            dst = cv2.imread(rng.choice(good), cv2.IMREAD_COLOR)
            if dst is None:
                continue
            res = cut_paste(img, msk, dst, cfg, rng)
            if res is None:
                continue
            syn, paste_mask = res
            name = f"{cls}_{stem}_{written:03d}.png"
            for folder, arr in ((out_dir, syn), (mask_dir, paste_mask)):
                tmp = os.path.join(folder, name + ".tmp.png")
                cv2.imwrite(tmp, arr)
                os.replace(tmp, os.path.join(folder, name))
            written += 1
            pbar.update(1)
        pbar.close()
        total_written += written
        if written < per_class:
            print(f"  ! {cls}: only {written}/{per_class} pastes landed on the part (rejected {attempts - written})")

    print(f"\n{total_written} synthetic images -> {out_root}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="override dataset.seed (which k real images are used)")
    args = ap.parse_args()
    cfg = common.apply_seed(common.load_config(), args.seed)
    common.ensure_dirs(cfg)
    print("=" * 70)
    print(f"Few-shot split + cut-paste synthetic defects   [seed {cfg['dataset']['seed']}]")
    print("=" * 70)
    split = make_split(cfg)
    print()
    generate_synthetic(cfg, split)
    print(f"\nNext: python scripts/3_refine_with_diffusion.py --seed {cfg['dataset']['seed']}")


if __name__ == "__main__":
    main()
