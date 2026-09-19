"""
Script 7: Export the stage-2 classifier as ONE ONNX graph: image -> class probabilities.

  python scripts/7_export_onnx.py                          # exports the best regime by held-out accuracy
  python scripts/7_export_onnx.py --regime real_plus_refined

The trained sklearn pipeline (StandardScaler + LogisticRegression) is folded into a single nn.Linear:
    softmax( W' x + b' )   with   W' = coef / scale,   b' = intercept - coef . (mean / scale)
That linear layer is attached to the ResNet50 backbone, so the exported model takes a normalised
224x224 image and returns probabilities. Validation compares ONNX output to sklearn.predict_proba on
real held-out images.

Outputs (results/):
  stage2_<regime>.onnx
  stage2_<regime>_model_card.json
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def fold_scaler_into_linear(scaler, clf):
    """
    Return (W, b) such that  logits = x @ W.T + b  equals  clf.decision_function(scaler.transform(x)).
    Works for multinomial LogisticRegression with >= 3 classes (coef_ shape (n_classes, n_features)).
    """
    coef = np.asarray(clf.coef_, dtype=np.float64)              # (K, D)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)    # (K,)
    mean = np.asarray(scaler.mean_, dtype=np.float64)           # (D,)
    scale = np.asarray(scaler.scale_, dtype=np.float64)         # (D,)
    if coef.shape[0] == 1:
        raise ValueError("Binary LogisticRegression (1 row of coef_) is not handled; this project uses >= 3 defect classes.")
    W = coef / scale[None, :]
    b = intercept - (coef * (mean / scale)[None, :]).sum(axis=1)
    return W.astype(np.float32), b.astype(np.float32)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


MODE = "full"


def pick_best_regime(res_dir):
    """Best regime by mean held-out accuracy over all seeds present."""
    import glob
    best, best_acc = None, -1
    for r in ("real_plus_refined", "real_plus_synthetic", "real_only"):
        files = [f for f in glob.glob(os.path.join(res_dir, f"stage2_{r}_s*.json"))
                 if (f.endswith(f"_{MODE}.json") if MODE != "full" else __import__("re").search(r"_s\d+\.json$", f))]
        if files:
            acc = float(np.mean([json.load(open(f))["accuracy"] for f in files]))
            if acc > best_acc:
                best, best_acc = r, acc
    if best is None:
        sys.exit("No stage-2 results found. Run script 5 first.")
    return best, best_acc


def build_export_module(W, b, cfg):
    import torch
    import torch.nn as nn
    from torchvision import models

    class ImageToDefectType(nn.Module):
        def __init__(self):
            super().__init__()
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            self.backbone = net
            self.head = nn.Linear(W.shape[1], W.shape[0])
            with torch.no_grad():
                self.head.weight.copy_(torch.from_numpy(W))
                self.head.bias.copy_(torch.from_numpy(b))

        def forward(self, x):                # x: (B, 3, H, W), ImageNet-normalised
            return torch.softmax(self.head(self.backbone(x)), dim=1)

    return ImageToDefectType().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=None)
    ap.add_argument("--seed", type=int, default=None, help="which seed's trained head to export (default: config seed)")
    ap.add_argument("--input", choices=["full", "crop"], default="full", help="which classifier variant to export")
    args = ap.parse_args()
    global MODE
    MODE = args.input

    cfg = common.apply_seed(common.load_config(), args.seed)
    res_dir, models_dir = cfg["paths"]["results_dir"], cfg["paths"]["models_dir"]
    regime, acc = (args.regime, None) if args.regime else pick_best_regime(res_dir)
    print("=" * 70)
    print(f"ONNX export: stage-2 classifier, regime = {regime}" + (f" (held-out acc {acc:.4f})" if acc else ""))
    print("=" * 70)

    tag = common.seed_tag(cfg) + ("" if MODE == "full" else f"_{MODE}")
    bundle = common.load_pickle(os.path.join(models_dir, f"stage2_{regime}_{tag}.pkl"))
    if MODE == "crop":
        print("note: the exported graph classifies a defect-centred crop; in deployment, crop around stage 1's most anomalous patch first.")
    scaler, clf, classes = bundle["scaler"], bundle["clf"], bundle["classes"]
    W, b = fold_scaler_into_linear(scaler, clf)
    print(f"folded head: Linear({W.shape[1]} -> {W.shape[0]})  classes = {classes}")

    import torch
    import onnx
    import onnxruntime as ort

    model = build_export_module(W, b, cfg)   # exported on CPU for portability
    size = cfg["features"]["input_size"]
    dummy = torch.randn(1, 3, size, size)
    onnx_path = os.path.join(res_dir, f"stage2_{regime}.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"], output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"exported -> {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # ---- validate: ONNX(image) vs sklearn(features(image)), SAME device, SAME preprocessing ----
    split = common.load_split(cfg)
    sample = [split["test"][c][0] for c in classes][:5]
    if MODE == "crop":
        # validate on defect-centred crops (what this head was trained on); GT centre is fine for a numeric check
        crops = []
        for p_ in sample:
            c = common.mask_centre_fraction(common.mask_path_for(p_, cfg)) or (0.5, 0.5)
            out_p = os.path.join(cfg["paths"]["cache_dir"], "crops", "onnx_val_" + os.path.basename(os.path.dirname(p_)) + "_" + os.path.basename(p_))
            crops.append(common.crop_around(p_, c, bundle.get("crop_frac") or cfg["classifier"]["crop_frac"], size, out_p))
        sample = crops

    fx_cpu = common.FeatureExtractor({**cfg, "features": {**cfg["features"], "device": "cpu"}})
    feats_cpu = fx_cpu.global_features(sample, desc="reference features (cpu, uncached)", use_cache=False)
    ref = clf.predict_proba(scaler.transform(feats_cpu))

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = torch.stack([fx_cpu._load(p) for p in sample]).numpy()
    out = sess.run(None, {"image": x})[0]
    max_abs = float(np.abs(out - ref).max())
    agree = int((out.argmax(1) == ref.argmax(1)).sum())
    ok = max_abs < 1e-3
    print(f"validation on {len(sample)} real {'crops' if MODE == 'crop' else 'images'} (same device): "
          f"max |onnx - sklearn| = {max_abs:.2e}, argmax agreement {agree}/{len(sample)}")
    print("numerical match: " + ("OK" if ok else "MISMATCH - do not deploy; check input normalisation"))

    # informational: how much do features drift between the training device (cached) and CPU inference?
    drift = None
    try:
        feats_cached = fx_cpu.global_features(sample, desc="cached features", use_cache=True)
        if np.any(feats_cached != feats_cpu):
            ref_cached = clf.predict_proba(scaler.transform(feats_cached))
            drift = float(np.abs(ref_cached - ref).max())
            print(f"cross-device drift (features cached from training device vs CPU): max prob diff {drift:.2e}; "
                  f"argmax agreement {int((ref_cached.argmax(1) == ref.argmax(1)).sum())}/{len(sample)} "
                  "(TF32 on NVIDIA GPUs is the usual cause; informational, not a deployment blocker)")
    except Exception:
        pass

    # ---- latency ----
    import time
    for _ in range(3):
        sess.run(None, {"image": x[:1]})
    t0 = time.perf_counter()
    for _ in range(10):
        sess.run(None, {"image": x[:1]})
    ms = (time.perf_counter() - t0) / 10 * 1000

    card = {
        "name": f"defect-type-classifier ({cfg['dataset']['name']})",
        "regime": regime, "classes": classes,
        "input": {"name": "image", "shape": ["batch", 3, size, size],
                  "preprocess": "RGB, resize to (size,size), scale to [0,1], normalise mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]"},
        "output": {"name": "probabilities", "shape": ["batch", len(classes)]},
        "backbone": "torchvision resnet50 IMAGENET1K_V1 (frozen)", "head": "StandardScaler + LogisticRegression folded into one Linear",
        "validation": {"n_images": len(sample), "max_abs_diff_vs_sklearn_same_device": max_abs, "passed": ok,
                       "cross_device_drift_max_prob_diff": drift},
        "cpu_latency_ms_batch1": round(ms, 1), "file_mb": round(os.path.getsize(onnx_path) / 1e6, 1),
        "intended_use": "runs on parts flagged by stage 1 (scripts/4_anomaly_stage.py); not a stand-alone good/bad detector",
        "classifier_input": MODE + ("" if MODE == "full" else f" (square crop, {bundle.get('crop_frac')} of image side, centred on stage-1's most anomalous patch)"),
    }
    card_path = os.path.join(res_dir, f"stage2_{regime}_model_card.json")
    json.dump(card, open(card_path, "w"), indent=2)
    print(f"model card -> {card_path}\nCPU latency (batch 1): {ms:.1f} ms")


if __name__ == "__main__":
    main()