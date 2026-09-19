"""
Script 3: Refine cut-paste synthetic defects with Stable Diffusion img2img.

  python scripts/3_refine_with_diffusion.py --limit 10     # tune first: look at results/diffusion_comparison/
  python scripts/3_refine_with_diffusion.py                # then run everything

Why: cut-paste leaves a boundary and a texture/lighting mismatch. Low-strength img2img harmonizes
the surface so the patch reads as part of one photo. The model's output is then composited back ONLY
inside the (slightly grown) paste region: the background and the rest of the part stay pixel-identical,
so diffusion cannot redraw the scene, and the defect stays where the class label says it is.

Input : data/synthetic/<category>/<class>/*.png
Output: data/synthetic_refined/<category>/<class>/*.png   (same filenames)
        results/diffusion_comparison/*.png                 (before | after pairs)

Runs on Intel Arc (xpu), NVIDIA (cuda) or CPU; see config.diffusion.device.
First run downloads ~4 GB of model weights and may spend a few minutes compiling kernels on Intel GPUs.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def load_pipeline(cfg, device):
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if device in ("cuda", "xpu") else torch.float32
    print(f"[diffusion] loading {cfg['diffusion']['model_id']} on {device} ({dtype})")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg["diffusion"]["model_id"],
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def paste_mask_for(in_path):
    """Script 2 saves the paste mask at <class>/_masks/<same filename>."""
    return os.path.join(os.path.dirname(in_path), "_masks", os.path.basename(in_path))


def composite_into_mask(original, refined, mask_path, cfg):
    """
    Combine the diffusion output with the raw cut-paste image.

    mode "seam" (default): the defect core keeps the RAW pasted pixels; the model's output is used only
        in a ring around the defect where the paste boundary is. Diffusion cannot heal the defect.
    mode "patch": the model's output replaces the whole (grown) paste region. Better texture harmony,
        but the model may smooth the defect toward a clean part. Use a low strength (<= 0.25).
    Everything outside the grown region is always the original pixel.
    """
    o = np.asarray(original.resize(refined.size), dtype=np.float32)
    r = np.asarray(refined, dtype=np.float32)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return refined
    m = cv2.resize(m, refined.size, interpolation=cv2.INTER_NEAREST)  # PIL size is (w, h), same as cv2 dsize
    m = (m > 0).astype(np.uint8)

    grow = int(cfg["diffusion"].get("mask_grow_px", 24))
    feather = int(cfg["diffusion"].get("mask_feather_px", 15)) | 1
    outer = cv2.dilate(m, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8)) if grow > 0 else m

    # colour-match the model output to the original inside the ring (SD drifts in hue on grey photos)
    ring = (outer > 0) & (m == 0)
    if ring.sum() > 100:
        mo, so = o[ring].mean(0), o[ring].std(0) + 1e-3
        mr, sr = r[ring].mean(0), r[ring].std(0) + 1e-3
        r = (r - mr) / sr * so + mo
        r = np.clip(r, 0, 255)

    a_outer = cv2.GaussianBlur(outer.astype(np.float32), (feather, feather), 0)[..., None]
    out = a_outer * r + (1 - a_outer) * o                       # model output inside grown region

    if cfg["diffusion"].get("refine_mode", "seam") == "seam":
        shrink = int(cfg["diffusion"].get("core_shrink_px", 3))
        core = cv2.erode(m, np.ones((shrink * 2 + 1, shrink * 2 + 1), np.uint8)) if shrink > 0 else m
        cf = int(cfg["diffusion"].get("core_feather_px", 5)) | 1
        a_core = cv2.GaussianBlur(core.astype(np.float32), (cf, cf), 0)[..., None]
        out = a_core * o + (1 - a_core) * out                   # raw defect pixels win in the core
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def refine_one(pipe, cfg, device, in_path, out_path, category, seed):
    import torch

    original = Image.open(in_path).convert("RGB")
    img = original.resize((512, 512), Image.LANCZOS)
    gen = torch.Generator(device=device if device != "xpu" else "cpu").manual_seed(seed)
    result = pipe(
        prompt=cfg["diffusion"]["prompt"].format(category=category.replace("_", " ")),
        negative_prompt=cfg["diffusion"]["negative_prompt"],
        image=img,
        strength=cfg["diffusion"]["strength"],
        guidance_scale=cfg["diffusion"]["guidance_scale"],
        num_inference_steps=cfg["diffusion"]["num_inference_steps"],
        generator=gen,
    ).images[0]
    if cfg["diffusion"].get("restrict_to_paste", True):
        result = composite_into_mask(img, result, paste_mask_for(in_path), cfg)
    else:
        result = result.resize(img.size)
    common.atomic_save(result, out_path)


def save_comparison(in_path, out_path, comp_dir):
    """raw | refined | refined with the paste region outlined in red (so faint defects can be located)"""
    W = 384
    a = Image.open(in_path).convert("RGB").resize((W, W))
    b = Image.open(out_path).convert("RGB").resize((W, W))
    c = np.asarray(b).copy()
    m = cv2.imread(paste_mask_for(in_path), cv2.IMREAD_GRAYSCALE)
    if m is not None:
        m = cv2.resize(m, (W, W), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(c, cnts, -1, (255, 0, 0), 2)
    c = Image.fromarray(c)
    canvas = Image.new("RGB", (W * 3 + 24, W + 28), "white")
    for i, im in enumerate((a, b, c)):
        canvas.paste(im, (i * (W + 12), 28))
    d = ImageDraw.Draw(canvas)
    for i, t in enumerate(("cut-paste (raw)", "diffusion refined", "paste region (red outline)")):
        d.text((i * (W + 12) + 8, 8), t, fill="black")
    canvas.save(os.path.join(comp_dir, os.path.basename(out_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="refine only the first N images (spread across classes)")
    ap.add_argument("--force", action="store_true", help="re-refine images that already have an output")
    ap.add_argument("--seed", type=int, default=None, help="must match the seed used in script 2")
    args = ap.parse_args()

    cfg = common.apply_seed(common.load_config(), args.seed)
    common.ensure_dirs(cfg)
    category = cfg["dataset"]["name"]
    src_root = common.synthetic_root(cfg)
    dst_root = common.synthetic_root(cfg, refined=True)

    # remove refined files whose raw counterpart no longer exists (stale from an earlier script-2 run)
    orphans = 0
    if os.path.isdir(dst_root):
        for cls in os.listdir(dst_root):
            d = os.path.join(dst_root, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(common.IMG_EXT) and not os.path.exists(os.path.join(src_root, cls, f)):
                    os.remove(os.path.join(d, f))
                    orphans += 1
    if orphans:
        print(f"removed {orphans} stale refined file(s) with no matching raw image")
    comp_dir = os.path.join(cfg["paths"]["results_dir"], "diffusion_comparison")
    Path(comp_dir).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(comp_dir):          # always show pairs from THIS run, never stale ones
        os.remove(os.path.join(comp_dir, f))

    if not os.path.isdir(src_root):
        sys.exit(f"No synthetic images at {src_root}. Run script 2 first.")

    # gather (class, file) jobs, interleaved across classes so --limit samples every class
    per_class = {c: common.list_images(os.path.join(src_root, c)) for c in sorted(os.listdir(src_root))}
    per_class = {c: v for c, v in per_class.items() if v}
    jobs = []
    for i in range(max(len(v) for v in per_class.values())):
        for c, v in per_class.items():
            if i < len(v):
                jobs.append((c, v[i]))
    if args.limit:
        jobs = jobs[: args.limit]
        print(f"[test mode] refining {len(jobs)} images; inspect {comp_dir} before running the full set")

    device = common.pick_device(cfg["diffusion"].get("device", "auto"))
    pipe = load_pipeline(cfg, device)

    done = skipped = 0
    for cls, in_path in tqdm(jobs, desc="refining"):
        out_dir = os.path.join(dst_root, cls)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(in_path))
        if os.path.exists(out_path) and not common.is_valid_image(out_path):
            os.remove(out_path)                  # truncated by an interrupted earlier run: redo it
        if os.path.exists(out_path) and not args.force:
            skipped += 1
            if done + skipped <= 12:
                save_comparison(in_path, out_path, comp_dir)   # still show the pair, so it is never stale
            continue
        seed = cfg["diffusion"]["seed"] + (abs(hash(os.path.basename(in_path))) % 100000)
        try:
            refine_one(pipe, cfg, device, in_path, out_path, category, seed)
            if done + skipped < 12:
                save_comparison(in_path, out_path, comp_dir)
            done += 1
        except Exception as e:  # keep going; report at the end
            print(f"\n  ! failed on {in_path}: {e}")

    print(f"\nrefined {done}, skipped {skipped} (already existed; use --force to redo) -> {dst_root}")
    print(f"before/after pairs -> {comp_dir}")
    if args.limit:
        print("\nHappy with the pairs? Re-run without --limit."
              "\n  seam still visible          -> raise mask_grow_px / mask_feather_px"
              "\n  defect looks smoothed/healed -> keep refine_mode: seam (default) or lower strength"
              "\n  ring looks different from the rest of the part -> lower strength to 0.25")
    else:
        print(f"\nNext: python scripts/5_train_classifier.py --regime all --seed {cfg['dataset']['seed']}"
              "   (and scripts/4_anomaly_stage.py once, if not done)")


if __name__ == "__main__":
    main()
