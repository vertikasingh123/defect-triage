"""
Script 4: Stage 1 - "is this part defective?"

  python scripts/4_anomaly_stage.py

A PatchCore-style detector: build a memory bank of mid-level ResNet patch descriptors from CLEAN
training images only, then score a test image by the largest nearest-neighbour distance among its
patches. No defect images are used here - that is the point: this stage needs zero labeled defects.

Follows the paper's recipe: wide_resnet50_2 layer2+layer3 patch descriptors, greedy k-center coreset
memory bank, max-of-min-distance image score. Input resolution is configurable; small defects on large
parts (e.g. screw) benefit from 320 px.

Outputs (results/):
  stage1_scores.npz          per-image scores, labels, paths
  stage1_summary.json        AUROC, threshold @ target TPR, what fraction of held-out defects is flagged
  stage1_roc.png
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def greedy_coreset(pool, n_select, device, proj_dim=128, seed=0):
    """
    PatchCore's k-center greedy selection on a random projection of the patches.
    Picks the patch farthest from everything selected so far, n_select times, so the bank
    covers the whole 'normal' manifold instead of piling up in its dense regions.
    """
    import torch
    from tqdm import tqdm
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.from_numpy(pool).to(device)                                   # (N, C)
    P = (torch.randn(X.shape[1], proj_dim, generator=g) / proj_dim ** 0.5).to(device)
    Xp = torch.empty((X.shape[0], proj_dim), device=device)
    for s_ in range(0, X.shape[0], 65536):
        Xp[s_:s_ + 65536] = X[s_:s_ + 65536] @ P
    n_select = min(n_select, X.shape[0])
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = int(torch.randint(0, X.shape[0], (1,), generator=g))
    min_d = ((Xp - Xp[selected[0]]) ** 2).sum(1)
    for i in tqdm(range(1, n_select), desc="greedy coreset", mininterval=1.0):
        selected[i] = int(torch.argmax(min_d))
        d = ((Xp - Xp[selected[i]]) ** 2).sum(1)
        min_d = torch.minimum(min_d, d)
    return pool[selected.cpu().numpy()]


def build_bank(fx, good_paths, cfg, rng):
    a = cfg["anomaly_stage"]
    layers = tuple(a["layers"])
    pool_cap = int(a.get("max_pool_patches", 200000))

    # extract in chunks so a high-resolution run does not need every patch in RAM at once
    chunks, total, per_img = [], 0, None
    step = 16
    for s_ in range(0, len(good_paths), step):
        feats = fx.patch_features(good_paths[s_:s_ + step], layer_names=layers, desc=f"bank {s_}/{len(good_paths)}")
        N, P, C = feats.shape
        per_img = P
        total += N * P
        chunks.append(feats.reshape(N * P, C))
    pool = np.concatenate(chunks, axis=0)
    del chunks
    if pool.shape[0] > pool_cap:                      # random pre-filter only if the pool is very large
        pool = pool[rng.choice(pool.shape[0], size=pool_cap, replace=False)]

    keep = min(int(a["bank_fraction"] * total), a["max_bank_patches"], pool.shape[0])
    if a.get("coreset", "greedy") == "greedy":
        bank = greedy_coreset(pool, keep, fx.device, seed=cfg["dataset"]["seed"])
    else:
        bank = pool[rng.choice(pool.shape[0], size=keep, replace=False)]
    print(f"memory bank: {bank.shape[0]} patches x {bank.shape[1]} dims "
          f"({a.get('coreset', 'greedy')} from {total} patches; {len(good_paths)} images x {per_img})")
    return bank


def score_images(fx, paths, bank, cfg, desc):
    """Image score = max over patches of (min L2 distance to any bank patch). Also returns the patch grid."""
    dists, grid = common.patch_min_distances(fx, paths, bank, cfg["anomaly_stage"]["layers"], desc=desc)
    return dists.max(1), grid


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = common.apply_seed(common.load_config(), args.seed)
    common.ensure_dirs(cfg)
    p = common.mvtec_paths(cfg)
    split = common.load_split(cfg)
    rng = np.random.default_rng(cfg["dataset"]["seed"])
    res_dir = cfg["paths"]["results_dir"]

    print("=" * 70)
    print("Stage 1: anomaly detection (PatchCore-style, good images only)")
    print("=" * 70)

    fx = common.FeatureExtractor(cfg,
                                 backbone=cfg["anomaly_stage"].get("backbone"),
                                 input_size=cfg["anomaly_stage"].get("input_size"))
    good_train = common.list_images(p["train_good"])
    bank = build_bank(fx, good_train, cfg, rng)

    # Evaluate on: test/good  vs  the HELD-OUT real defects (never the k-shot training ones)
    test_good = split["test_good"]
    test_bad = [path for cls in split["classes"] for path in split["test"][cls]]
    paths = test_good + test_bad
    labels = np.array([0] * len(test_good) + [1] * len(test_bad))
    scores, grid = score_images(fx, paths, bank, cfg, desc="scoring test images")
    common.save_bank(cfg, bank, grid, {"backbone": cfg["anomaly_stage"].get("backbone", "resnet50"),
                                       "input_size": cfg["anomaly_stage"].get("input_size", cfg["features"]["input_size"]),
                                       "layers": list(cfg["anomaly_stage"]["layers"])})

    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    # threshold at the target TPR: the smallest score that still flags target_tpr of real defects
    target = cfg["anomaly_stage"]["target_tpr"]
    i = int(np.argmax(tpr >= target))
    threshold = float(thr[i])
    flagged = scores >= threshold
    defect_recall = float(flagged[labels == 1].mean())
    good_fpr = float(flagged[labels == 0].mean())

    print(f"\nimage-level AUROC : {auroc:.4f}")
    print(f"threshold @TPR>={target:.2f}: {threshold:.4f}")
    print(f"  held-out defects flagged : {defect_recall*100:.1f}%  ({int(flagged[labels==1].sum())}/{len(test_bad)})")
    print(f"  good parts falsely flagged: {good_fpr*100:.1f}%  ({int(flagged[labels==0].sum())}/{len(test_good)})")

    np.savez(os.path.join(res_dir, "stage1_scores.npz"),
             scores=scores, labels=labels, paths=np.array(paths), threshold=threshold)
    with open(os.path.join(res_dir, "stage1_summary.json"), "w") as f:
        json.dump({"auroc": auroc, "threshold": threshold, "target_tpr": target,
                   "defect_recall": defect_recall, "good_fpr": good_fpr,
                   "n_test_good": len(test_good), "n_test_defect": len(test_bad),
                   "bank_patches": int(bank.shape[0]), "bank_dims": int(bank.shape[1])}, f, indent=2)

    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, lw=2.5, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.scatter([fpr[i]], [tpr[i]], s=60, zorder=5, label=f"operating point (TPR {tpr[i]:.2f}, FPR {fpr[i]:.2f})")
    plt.xlabel("False positive rate (good parts flagged)")
    plt.ylabel("True positive rate (defects flagged)")
    plt.title(f"Stage 1 - {p['category']}: defective vs good")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "stage1_roc.png"), dpi=200)
    print(f"\nsaved -> {res_dir}/stage1_summary.json, stage1_scores.npz, stage1_roc.png, {common.bank_path(cfg)}")
    print("\nNext: python scripts/5_train_classifier.py --regime all")


if __name__ == "__main__":
    main()
