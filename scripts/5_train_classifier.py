"""
Script 5: Stage 2 - "what kind of defect is it?"  (few-shot, k real images per class)

  python scripts/5_train_classifier.py --regime all
  python scripts/5_train_classifier.py --regime real_only
  python scripts/5_train_classifier.py --regime real_plus_synthetic
  python scripts/5_train_classifier.py --regime real_plus_refined

Every regime uses the SAME k real training images per class and the SAME classical augmentation
(8 flip/rotate variants). The regimes differ only in what extra data is added:

  real_only            : k real (x8 dihedral)
  real_plus_synthetic  : + cut-paste synthetic (raw)
  real_plus_refined    : + cut-paste synthetic refined by Stable Diffusion

Classifier: frozen ResNet50 features -> StandardScaler -> multinomial LogisticRegression.
Evaluated on the HELD-OUT real defect images (never seen by anything above).

Input to the classifier (config classifier.input):
  "full"   : the whole image resized to 224 px (small defects nearly vanish)
  "crop"   : a square crop centred on the defect, so small defects are seen at higher resolution.
             Training crops are centred on the known mask (GT mask for real, paste mask for synthetic).
             TEST crops are centred on the most anomalous patch found by STAGE 1 (no ground truth used),
             which is how the deployed system would work. Requires models/stage1_bank.npz from script 4.
  "oracle" : like crop, but test crops use the GT mask centre. Upper bound for the crop idea; not deployable.

Outputs:
  models/stage2_<regime>.pkl
  results/stage2_<regime>.json    (accuracy, macro-F1, per-class recall, confusion matrix, predictions)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

REGIMES = ["real_only", "real_plus_synthetic", "real_plus_refined"]


def dihedral_variants(path, cache_dir):
    """Write the 8 flip/rotate variants of an image to disk once; return their paths."""
    out_dir = os.path.join(cache_dir, "dihedral")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    cls = os.path.basename(os.path.dirname(path))
    paths = []
    img = None
    for k in range(8):
        out = os.path.join(out_dir, f"{cls}_{stem}_d{k}.png")
        if os.path.exists(out) and not common.is_valid_image(out):
            os.remove(out)                       # truncated by an interrupted earlier run
        if not os.path.exists(out):
            if img is None:
                img = Image.open(path).convert("RGB")
            v = img.rotate(90 * (k % 4), expand=True)
            if k >= 4:
                v = v.transpose(Image.FLIP_LEFT_RIGHT)
            common.atomic_save(v, out)
        paths.append(out)
    return paths


def crop_out_path(cfg, image_path, tag):
    import hashlib
    h = hashlib.md5(f"{os.path.abspath(image_path)}|{tag}".encode()).hexdigest()[:16]
    cls = os.path.basename(os.path.dirname(image_path))
    return os.path.join(cfg["paths"]["cache_dir"], "crops", f"{cls}_{os.path.splitext(os.path.basename(image_path))[0]}_{h}.png")


def train_crop(cfg, image_path, mask_path):
    """Crop a training image around its known defect mask (GT or paste mask)."""
    c = common.mask_centre_fraction(mask_path)
    if c is None:
        c = (0.5, 0.5)
    return common.crop_around(image_path, c, cfg["classifier"]["crop_frac"], cfg["features"]["input_size"],
                              crop_out_path(cfg, image_path, f"train_{cfg['classifier']['crop_frac']}"))


def test_crops(cfg, test_paths, mode):
    """Crop test images around the defect: located by stage 1 ('crop') or by the GT mask ('oracle')."""
    outs = []
    if mode == "oracle":
        for p_ in test_paths:
            c = common.mask_centre_fraction(common.mask_path_for(p_, cfg)) or (0.5, 0.5)
            outs.append(common.crop_around(p_, c, cfg["classifier"]["crop_frac"], cfg["features"]["input_size"],
                                           crop_out_path(cfg, p_, f"oracle_{cfg['classifier']['crop_frac']}")))
        return outs, None

    bank, grid, meta = common.load_bank(cfg)
    fx1 = common.FeatureExtractor(cfg, backbone=meta["backbone"], input_size=meta["input_size"])
    dists, grid = common.patch_min_distances(fx1, test_paths, bank, meta["layers"], desc="stage-1 localisation")
    hits = 0
    for p_, row in zip(test_paths, dists):
        c = common.localize_from_distances(row, grid)
        outs.append(common.crop_around(p_, c, cfg["classifier"]["crop_frac"], cfg["features"]["input_size"],
                                       crop_out_path(cfg, p_, f"stage1_{cfg['classifier']['crop_frac']}")))
        gt = common.mask_centre_fraction(common.mask_path_for(p_, cfg))
        if gt is not None:
            # count as a hit if the GT centre falls inside the crop
            half = cfg["classifier"]["crop_frac"] / 2
            hits += int(abs(gt[0] - c[0]) <= half and abs(gt[1] - c[1]) <= half)
    loc_acc = hits / len(test_paths)
    print(f"  stage-1 localisation: GT defect centre inside the crop for {hits}/{len(test_paths)} test images ({loc_acc:.1%})")
    return outs, loc_acc


def gather_training_set(cfg, split, regime):
    """Return (paths, labels, composition dict)."""
    classes = split["classes"]
    cat = cfg["dataset"]["name"]
    paths, labels = [], []
    comp = {}

    mode = cfg["classifier"].get("input", "full")
    for ci, cls in enumerate(classes):
        real = split["train"][cls]
        if mode in ("crop", "oracle"):
            real = [train_crop(cfg, pth, common.mask_path_for(pth, cfg)) for pth in real]
        if cfg["classifier"].get("dihedral_aug", True):
            real_aug = [q for pth in real for q in dihedral_variants(pth, cfg["paths"]["cache_dir"])]
        else:
            real_aug = list(real)
        paths += real_aug
        labels += [ci] * len(real_aug)
        comp.setdefault("real (with aug)", 0)
        comp["real (with aug)"] += len(real_aug)

        if regime == "real_plus_synthetic":
            syn = common.list_images(os.path.join(common.synthetic_root(cfg), cls))
            if mode in ("crop", "oracle"):
                syn = [train_crop(cfg, q, os.path.join(os.path.dirname(q), "_masks", os.path.basename(q))) for q in syn]
            paths += syn
            labels += [ci] * len(syn)
            comp["cut-paste synthetic"] = comp.get("cut-paste synthetic", 0) + len(syn)
        elif regime == "real_plus_refined":
            ref = common.list_images(os.path.join(common.synthetic_root(cfg, refined=True), cls))
            if not ref:
                sys.exit(f"No refined images for class '{cls}' (seed {cfg['dataset']['seed']}). Run script 3 without --limit first.")
            if mode in ("crop", "oracle"):
                # refined images share filenames with the raw synthetic ones, whose paste masks we reuse
                ref = [train_crop(cfg, q, os.path.join(common.synthetic_root(cfg), cls, "_masks", os.path.basename(q))) for q in ref]
            paths += ref
            labels += [ci] * len(ref)
            comp["diffusion-refined synthetic"] = comp.get("diffusion-refined synthetic", 0) + len(ref)

    return paths, np.array(labels), comp


def gather_test_set(split):
    paths, labels = [], []
    for ci, cls in enumerate(split["classes"]):
        paths += split["test"][cls]
        labels += [ci] * len(split["test"][cls])
    return paths, np.array(labels)


def train_and_eval(fx, cfg, split, regime, test_paths, test_labels, X_test_cache, mode="full", loc_acc=None):
    classes = split["classes"]
    train_paths, y_train, comp = gather_training_set(cfg, split, regime)

    print(f"\n--- regime: {regime} ---")
    for k, v in comp.items():
        print(f"  {k:<30}{v:>6} images")
    print(f"  {'total train':<30}{len(train_paths):>6}")
    print(f"  {'held-out real test':<30}{len(test_paths):>6}")

    X_train = fx.global_features(train_paths, desc=f"train features [{regime}]")
    if X_test_cache["X"] is None:
        X_test_cache["X"] = fx.global_features(test_paths, desc="test features")
    X_test = X_test_cache["X"]

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=cfg["classifier"]["C"], max_iter=5000, random_state=cfg["dataset"]["seed"])
    clf.fit(scaler.transform(X_train), y_train)

    proba = clf.predict_proba(scaler.transform(X_test))
    pred = proba.argmax(1)

    acc = accuracy_score(test_labels, pred)
    macro_f1 = f1_score(test_labels, pred, average="macro")
    per_class_recall = recall_score(test_labels, pred, average=None, labels=list(range(len(classes))))
    cm = confusion_matrix(test_labels, pred, labels=list(range(len(classes))))

    print(f"\n  accuracy  : {acc:.4f}")
    print(f"  macro-F1  : {macro_f1:.4f}")
    for cls, r in zip(classes, per_class_recall):
        print(f"    recall[{cls}] = {r:.3f}")

    tag = common.seed_tag(cfg) + ("" if mode == "full" else f"_{mode}")
    common.save_pickle({"scaler": scaler, "clf": clf, "classes": classes, "regime": regime, "seed": cfg["dataset"]["seed"],
                        "input": mode, "crop_frac": cfg["classifier"].get("crop_frac")},
                       os.path.join(cfg["paths"]["models_dir"], f"stage2_{regime}_{tag}.pkl"))
    with open(os.path.join(cfg["paths"]["results_dir"], f"stage2_{regime}_{tag}.json"), "w") as f:
        json.dump({
            "regime": regime, "seed": cfg["dataset"]["seed"], "input": mode, "localisation_acc": loc_acc,
            "classes": classes, "composition": comp,
            "n_train": len(train_paths), "n_test": len(test_paths),
            "accuracy": acc, "macro_f1": macro_f1,
            "per_class_recall": dict(zip(classes, map(float, per_class_recall))),
            "confusion_matrix": cm.tolist(),
            "test_paths": test_paths, "test_labels": test_labels.tolist(),
            "pred": pred.tolist(), "proba": proba.tolist(),
        }, f, indent=1)
    return acc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=REGIMES + ["all"], default="all")
    ap.add_argument("--seed", type=int, default=None, help="must match the seed used in scripts 2 and 3")
    ap.add_argument("--input", choices=["full", "crop", "oracle"], default=None, help="override classifier.input")
    args = ap.parse_args()

    cfg = common.apply_seed(common.load_config(), args.seed)
    if args.input:
        cfg["classifier"]["input"] = args.input
    common.ensure_dirs(cfg)
    split = common.load_split(cfg)
    regimes = REGIMES if args.regime == "all" else [args.regime]

    print("=" * 70)
    print(f"Stage 2: few-shot defect type classification  (k={split['k_shot']}, {len(split['classes'])} classes, seed {cfg['dataset']['seed']})")
    print("=" * 70)

    fx = common.FeatureExtractor(cfg)
    test_paths, test_labels = gather_test_set(split)
    mode = cfg["classifier"].get("input", "full")
    loc_acc = None
    if mode in ("crop", "oracle"):
        print(f"\nclassifier input: '{mode}' crops of {cfg['classifier']['crop_frac']:.0%} of the image around the defect")
        test_paths, loc_acc = test_crops(cfg, test_paths, mode)
    X_test_cache = {"X": None}

    summary = {}
    for r in regimes:
        summary[r] = train_and_eval(fx, cfg, split, r, test_paths, test_labels, X_test_cache, mode, loc_acc)

    print("\n" + "=" * 70)
    print(f"{'regime':<24}{'accuracy':>10}{'macro-F1':>10}")
    for r, (acc, f1) in summary.items():
        print(f"{r:<24}{acc:>10.4f}{f1:>10.4f}")
    print("=" * 70)
    print("\nNext: python scripts/6_evaluate.py")


if __name__ == "__main__":
    main()
