"""
Script 6: Compare the three stage-2 regimes and write the report.

  python scripts/6_evaluate.py

Reads results/stage2_<regime>.json (and stage1_summary.json if present). Writes to results/:
  fig1_regime_comparison.png     accuracy + macro-F1 per regime
  fig2_per_class_recall.png      which defect types benefit from synthetic / refined data
  fig3_confusion_matrices.png    one confusion matrix per regime
  summary.json, REPORT.md
"""

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

REGIMES = ["real_only", "real_plus_synthetic", "real_plus_refined"]
LABELS = {"real_only": "Real only\n(k-shot + flips/rots)",
          "real_plus_synthetic": "+ cut-paste\nsynthetic",
          "real_plus_refined": "+ diffusion-refined\nsynthetic"}
COLORS = {"real_only": "#7f8c8d", "real_plus_synthetic": "#e67e22", "real_plus_refined": "#2980b9"}


import glob


import re


def result_files(res_dir, regime, mode):
    """stage2_<regime>_s<seed>.json for mode 'full'; stage2_<regime>_s<seed>_<mode>.json otherwise."""
    pat = re.compile(rf"stage2_{regime}_s\d+" + ("" if mode == "full" else f"_{mode}") + r"\.json$")
    return sorted(f for f in glob.glob(os.path.join(res_dir, f"stage2_{regime}_s*.json")) if pat.search(os.path.basename(f)))


def load(res_dir, mode="full", quiet=False):
    """
    Returns R[regime] = a dict shaped like one script-5 result but with metrics AVERAGED over seeds,
    plus R[regime]["seeds"] = list of per-seed dicts. With one seed this is identical to that seed.
    """
    out = {}
    for r in REGIMES:
        files = result_files(res_dir, r, mode)
        if not files:
            if not quiet:
                print(f"  (missing) stage2_{r}_s*{'' if mode == 'full' else '_' + mode}.json")
            continue
        seeds = [json.load(open(f)) for f in files]
        classes = seeds[0]["classes"]
        acc = np.array([d["accuracy"] for d in seeds])
        f1 = np.array([d["macro_f1"] for d in seeds])
        rec = {c: np.array([d["per_class_recall"][c] for d in seeds]) for c in classes}
        cm = np.mean([np.array(d["confusion_matrix"], dtype=float) for d in seeds], axis=0)
        out[r] = {
            "regime": r, "classes": classes, "n_seeds": len(seeds),
            "seed_list": [d.get("seed") for d in seeds],
            "accuracy": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            "macro_f1": float(f1.mean()), "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
            "per_class_recall": {c: float(v.mean()) for c, v in rec.items()},
            "per_class_recall_std": {c: (float(v.std(ddof=1)) if len(v) > 1 else 0.0) for c, v in rec.items()},
            "confusion_matrix": cm.tolist(),
            "composition": seeds[0]["composition"], "n_train": seeds[0]["n_train"], "n_test": seeds[0]["n_test"],
            "input": seeds[0].get("input", "full"),
            "localisation_acc": float(np.mean([d["localisation_acc"] for d in seeds if d.get("localisation_acc") is not None]))
                                if any(d.get("localisation_acc") is not None for d in seeds) else None,
            "seeds": seeds,
        }
    if not out and not quiet:
        sys.exit(f"No stage-2 results for input mode '{mode}'. Run: python scripts/5_train_classifier.py --regime all"
                 + ("" if mode == "full" else f" --input {mode}"))
    n = {r: out[r]["n_seeds"] for r in out}
    if len(set(n.values())) > 1:
        print(f"  ! regimes have different numbers of seeds: {n}")
    return out


def fig_regime_comparison(R, res_dir):
    regs = [r for r in REGIMES if r in R]
    acc = [R[r]["accuracy"] for r in regs]
    f1 = [R[r]["macro_f1"] for r in regs]
    x = np.arange(len(regs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    acc_e = [R[r]["accuracy_std"] for r in regs]
    f1_e = [R[r]["macro_f1_std"] for r in regs]
    b1 = ax.bar(x - w / 2, acc, w, yerr=acc_e, capsize=4, label="accuracy", color=[COLORS[r] for r in regs], alpha=0.9)
    b2 = ax.bar(x + w / 2, f1, w, yerr=f1_e, capsize=4, label="macro-F1", color=[COLORS[r] for r in regs], alpha=0.5, hatch="//")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in regs])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score on held-out real defects")
    ns = R[regs[0]]["n_seeds"]
    ax.set_title(f"Stage 2: defect-type classification, k={_k(R)} real images per class"
                 + (f"  (mean ± sd over {ns} seeds)" if ns > 1 else ""))
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, f"fig1_regime_comparison{_sfx(R)}.png"), dpi=200)
    plt.close(fig)


def _sfx(R):
    m = next(iter(R.values())).get("input", "full")
    return "" if m == "full" else f"_{m}"


def _k(R):
    # infer k from the real_only composition (real with aug / 8 / n_classes) if available
    r = R.get("real_only") or next(iter(R.values()))
    n_real = r["composition"].get("real (with aug)", 0)
    n_cls = len(r["classes"])
    return int(round(n_real / 8 / n_cls)) if n_real and n_cls else "?"


def fig_per_class_recall(R, res_dir):
    regs = [r for r in REGIMES if r in R]
    classes = R[regs[0]]["classes"]
    x = np.arange(len(classes))
    w = 0.8 / len(regs)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(classes)), 4.6))
    for i, r in enumerate(regs):
        vals = [R[r]["per_class_recall"][c] for c in classes]
        errs = [R[r]["per_class_recall_std"][c] for c in classes]
        ax.bar(x + (i - (len(regs) - 1) / 2) * w, vals, w, yerr=errs, capsize=3,
               label=LABELS[r].replace("\n", " "), color=COLORS[r])
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall (held-out real)")
    ax.set_title("Per-class recall: where does synthetic / refined data help?")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, f"fig2_per_class_recall{_sfx(R)}.png"), dpi=200)
    plt.close(fig)


def fig_confusions(R, res_dir):
    regs = [r for r in REGIMES if r in R]
    classes = R[regs[0]]["classes"]
    short = [c[:12] for c in classes]
    fig, axes = plt.subplots(1, len(regs), figsize=(5 * len(regs), 4.6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, regs):
        cm = np.array(R[r]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        fmt = (lambda v: f"{v:.0f}") if R[r]["n_seeds"] == 1 else (lambda v: f"{v:.1f}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, fmt(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{LABELS[r].replace(chr(10), ' ')}\nacc={R[r]['accuracy']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, f"fig3_confusion_matrices{_sfx(R)}.png"), dpi=200)
    plt.close(fig)


def write_report(R, stage1, cfg, res_dir, others=None):
    regs = [r for r in REGIMES if r in R]
    mode = R[regs[0]].get("input", "full")
    base = R.get("real_only")
    cat = cfg["dataset"]["name"]
    k = _k(R)
    L = []
    L.append(f"# Two-stage defect triage on MVTec AD / {cat}\n")
    L.append("**Question.** A one-class detector can flag a bad part with zero labeled defects, but it cannot say "
             "*what kind* of defect it is, and root-cause analysis needs the type. With only a handful of real "
             "examples per defect type, does adding cut-paste synthetic examples help a type classifier, and does "
             "refining those synthetics with Stable Diffusion help further?\n")

    if stage1:
        L.append("## Stage 1 - is it defective? (good images only)\n")
        L.append(f"- image-level AUROC: **{stage1['auroc']:.4f}**")
        L.append(f"- at threshold {stage1['threshold']:.3f} (chosen for TPR >= {stage1['target_tpr']:.2f}): "
                 f"{stage1['defect_recall']*100:.1f}% of held-out defects flagged, "
                 f"{stage1['good_fpr']*100:.1f}% of good parts falsely flagged "
                 f"(n = {stage1['n_test_defect']} defects, {stage1['n_test_good']} good)")
        a = cfg["anomaly_stage"]
        L.append(f"- memory bank: {stage1['bank_patches']} patches x {stage1['bank_dims']} dims "
                 f"({a.get('coreset', 'greedy')} coreset; {a.get('backbone', 'resnet50')} {'+'.join(a['layers'])} at {a.get('input_size', cfg['features']['input_size'])} px)\n")

    ns = R[regs[0]]["n_seeds"]
    mode_txt = {"full": "whole image at 224 px",
                "crop": f"{cfg['classifier']['crop_frac']:.0%} crop around the defect located by STAGE 1 (no ground truth at test time)",
                "oracle": f"{cfg['classifier']['crop_frac']:.0%} crop around the ground-truth defect centre (upper bound, not deployable)"}[mode]
    L.append(f"Classifier input: **{mode_txt}**." +
             (f" Stage 1 placed the true defect centre inside the crop for {R[regs[0]]['localisation_acc']:.1%} of test images."
              if R[regs[0]].get("localisation_acc") is not None else "") + "\n")
    L.append(f"## Stage 2 - what kind of defect? (k = {k} real images per class"
             + (f", mean ± sd over {ns} seeds {R[regs[0]]['seed_list']})" if ns > 1 else ", single seed)") + "\n")
    L.append("| regime | extra training data | accuracy | macro-F1 | delta acc vs real-only |")
    L.append("|---|---|---:|---:|---:|")
    for r in regs:
        extra = ", ".join(f"{v} {kk}" for kk, v in R[r]["composition"].items() if kk != "real (with aug)") or "-"
        d = R[r]["accuracy"] - base["accuracy"] if base else float("nan")
        pm = (lambda m, sd: f"{m:.3f} ± {sd:.3f}") if ns > 1 else (lambda m, sd: f"{m:.4f}")
        L.append(f"| {r} | {extra} | {pm(R[r]['accuracy'], R[r]['accuracy_std'])} | "
                 f"{pm(R[r]['macro_f1'], R[r]['macro_f1_std'])} | {d:+.4f} |")
    L.append("")
    if ns > 1:
        L.append("Per-seed accuracy:\n")
        L.append("| seed | " + " | ".join(regs) + " |")
        L.append("|---|" + "---:|" * len(regs))
        all_seeds = sorted({a.get("seed") for r in regs for a in R[r]["seeds"]}, key=lambda x: (x is None, x))
        for sd in all_seeds:
            cells = []
            for r in regs:
                m = {a.get("seed"): a["accuracy"] for a in R[r]["seeds"]}
                cells.append(f"{m[sd]:.4f}" if sd in m else "-")
            L.append(f"| {sd} | " + " | ".join(cells) + " |")
        L.append("")

    classes = R[regs[0]]["classes"]
    L.append("Per-class recall on held-out real defects:\n")
    L.append("| class | " + " | ".join(regs) + " |")
    L.append("|---|" + "---:|" * len(regs))
    for c in classes:
        L.append(f"| {c} | " + " | ".join(f"{R[r]['per_class_recall'][c]:.3f}" for r in regs) + " |")
    L.append("")

    # plain-language reading of the numbers, written conditionally so it never overclaims.
    # With several seeds the verdict comes from the PAIRED per-seed differences (same split, same test set),
    # not from the difference of means: a sign that flips between seeds is 'neutral' whatever the mean says.
    L.append("## Reading the result\n")

    def verdict(a, b, label_pos, label_neg, label_neutral):
        pa = {x.get("seed"): x["accuracy"] for x in R[a]["seeds"]}
        pb = {x.get("seed"): x["accuracy"] for x in R[b]["seeds"]}
        common_s = sorted(set(pa) & set(pb))
        d = np.array([pa[sd] - pb[sd] for sd in common_s])
        if len(d) == 0:
            return None
        mean = float(d.mean())
        if len(d) > 1:
            consistent = bool((d > 0).all() or (d < 0).all())
            if consistent and mean > 0.01:
                return f"{label_pos} {mean:+.3f} accuracy (positive in {len(d)}/{len(d)} seeds)."
            if consistent and mean < -0.01:
                return f"{label_neg} {mean:+.3f} accuracy (negative in {len(d)}/{len(d)} seeds)."
            return f"{label_neutral} ({mean:+.3f}; sign {'consistent but small' if consistent else 'not consistent across seeds'})."
        if mean > 0.02:
            return f"{label_pos} {mean:+.3f} accuracy (single seed)."
        if mean < -0.02:
            return f"{label_neg} {mean:+.3f} accuracy (single seed)."
        return f"{label_neutral} ({mean:+.3f}, single seed)."

    if "real_plus_synthetic" in R and base:
        v = verdict("real_plus_synthetic", "real_only",
                    "- Cut-paste synthetic data **helps**:", "- Cut-paste synthetic data **hurts**:",
                    "- Cut-paste synthetic data is **neutral**")
        if v:
            L.append(v)
    if "real_plus_refined" in R and "real_plus_synthetic" in R:
        v = verdict("real_plus_refined", "real_plus_synthetic",
                    "- Diffusion (seam) refinement **adds** over raw cut-paste:",
                    "- Diffusion (seam) refinement **reduces** accuracy vs raw cut-paste:",
                    "- Diffusion (seam) refinement gives **no measurable benefit** over raw cut-paste")
        if v:
            L.append(v)
    L.append(f"- Held-out set: {R[regs[0]]['n_test']} real defect images the classifier never saw in any form "
             f"(one standard error on accuracy at this n is about ±{(0.25 / R[regs[0]]['n_test']) ** 0.5:.3f}).")
    if ns > 1 and "real_plus_refined" in R and "real_plus_synthetic" in R:
        by_seed = lambda r: {a.get("seed"): a["accuracy"] for a in R[r]["seeds"]}
        rf, rs, ro = by_seed("real_plus_refined"), by_seed("real_plus_synthetic"), by_seed("real_only")
        common_seeds = sorted(set(rf) & set(rs))
        d = np.array([rf[sd] - rs[sd] for sd in common_seeds])
        if len(d):
            L.append(f"- Refined minus raw cut-paste, per seed {common_seeds}: {', '.join(f'{v:+.3f}' for v in d)} "
                     f"(mean {d.mean():+.3f}). " + ("Consistent sign across seeds." if (d > 0).all() or (d < 0).all()
                                                     else "Sign is NOT consistent across seeds; treat as neutral."))
        common2 = sorted(set(rs) & set(ro))
        d2 = np.array([rs[sd] - ro[sd] for sd in common2])
        if len(d2):
            L.append(f"- Raw cut-paste minus real-only, per seed {common2}: {', '.join(f'{v:+.3f}' for v in d2)} "
                     f"(mean {d2.mean():+.3f}). " + ("Consistent sign across seeds." if (d2 > 0).all() or (d2 < 0).all()
                                                      else "Sign is NOT consistent across seeds."))
    else:
        L.append("- Caveat: single seed. k-shot results vary a lot with which k images are drawn; "
                 "run scripts 2-3-5 with `--seed 1`, `--seed 2` and re-run this script before quoting a number.")
    L.append("")

    if others:
        L.append("## Whole image vs. defect crop\n")
        L.append("| classifier input | " + " | ".join(regs) + " | thread_side recall (refined) |")
        L.append("|---|" + "---:|" * (len(regs) + 1))
        for name, RR in [(mode, R)] + others:
            if not RR:
                continue
            row = " | ".join(f"{RR[r]['accuracy']:.3f}" if r in RR else "-" for r in regs)
            ts = RR.get("real_plus_refined", {}).get("per_class_recall", {}).get("thread_side")
            L.append(f"| {name} | {row} | {ts:.3f} |" if ts is not None else f"| {name} | {row} | - |")
        L.append("")
        L.append("Reading: if 'crop' is well above 'full', small defects were a resolution problem, and stage 1's "
                 "localisation is good enough to solve it in deployment. 'oracle' shows the ceiling if localisation were perfect.\n")
    L.append("## Method\n")
    a = cfg["anomaly_stage"]
    L.append(f"- Stage 1: PatchCore recipe - {a.get('backbone', 'resnet50')} {'+'.join(a['layers'])} patch descriptors at "
             f"{a.get('input_size', cfg['features']['input_size'])} px, {a.get('coreset', 'greedy')} k-center coreset memory bank "
             f"built from good images only, image score = max over patches of nearest-bank distance.")
    L.append(f"- Stage 2 features: frozen ImageNet {cfg['features'].get('backbone', 'resnet50')}, global pooled (2048-d) at {cfg['features']['input_size']} px.")
    L.append(f"- Stage 2: StandardScaler + multinomial LogisticRegression (C={cfg['classifier']['C']}). "
             f"All regimes see the same k real images with 8 flip/rotate variants; regimes differ only in added synthetic data.")
    L.append(f"- Synthetic: defect region cut from each real training image using MVTec's ground-truth mask, pasted onto a random clean image "
             f"with +/-{cfg['synthetic']['rotate_deg']} deg rotation, scale {cfg['synthetic']['scale_range']}, feathered edge.")
    mode = cfg["diffusion"].get("refine_mode", "seam")
    where = ("only the ring around the paste is repainted; the defect's own pixels are kept" if mode == "seam"
             else "the whole paste region including the defect is repainted")
    L.append(f"- Refinement: Stable Diffusion img2img (strength {cfg['diffusion']['strength']}, "
             f"{cfg['diffusion']['num_inference_steps']} steps, guidance {cfg['diffusion']['guidance_scale']}); "
             f"mode '{mode}': {where}. Everything outside a {cfg['diffusion'].get('mask_grow_px', 24)} px ring is the original pixel.\n")
    L.append("Figures: `fig1_regime_comparison.png`, `fig2_per_class_recall.png`, `fig3_confusion_matrices.png`, `stage1_roc.png`.")

    with open(os.path.join(res_dir, f"REPORT{_sfx(R)}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    summary = {r: {"accuracy": R[r]["accuracy"], "accuracy_std": R[r]["accuracy_std"],
                   "macro_f1": R[r]["macro_f1"], "macro_f1_std": R[r]["macro_f1_std"],
                   "n_seeds": R[r]["n_seeds"], "seeds": R[r]["seed_list"],
                   "per_class_recall": R[r]["per_class_recall"], "composition": R[r]["composition"]} for r in regs}
    if stage1:
        summary["stage1"] = stage1
    with open(os.path.join(res_dir, f"summary{_sfx(R)}.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["full", "crop", "oracle"], default="full",
                    help="which stage-2 results to report (others are shown in a comparison table if present)")
    args = ap.parse_args()
    cfg = common.load_config()
    res_dir = cfg["paths"]["results_dir"]
    print("=" * 70)
    print(f"Evaluation  (classifier input: {args.input})")
    print("=" * 70)
    R = load(res_dir, args.input)
    others = [(m, load(res_dir, m, quiet=True)) for m in ("full", "crop", "oracle") if m != args.input]
    others = [(m, RR) for m, RR in others if RR]
    s1p = os.path.join(res_dir, "stage1_summary.json")
    stage1 = json.load(open(s1p)) if os.path.exists(s1p) else None

    fig_regime_comparison(R, res_dir)
    fig_per_class_recall(R, res_dir)
    fig_confusions(R, res_dir)
    write_report(R, stage1, cfg, res_dir, others)

    ns = max(R[r]["n_seeds"] for r in R)
    print(f"\n{'regime':<24}{'accuracy':>16}{'macro-F1':>16}   ({ns} seed{'s' if ns > 1 else ''})")
    for r in REGIMES:
        if r in R:
            a = f"{R[r]['accuracy']:.4f}" + (f" ± {R[r]['accuracy_std']:.3f}" if ns > 1 else "")
            f = f"{R[r]['macro_f1']:.4f}" + (f" ± {R[r]['macro_f1_std']:.3f}" if ns > 1 else "")
            print(f"{r:<24}{a:>16}{f:>16}")
    print(f"\nwrote -> {res_dir}/REPORT{_sfx(R)}.md, summary{_sfx(R)}.json, fig1-3*{_sfx(R)}.png")
    print("\nNext: python scripts/7_export_onnx.py")


if __name__ == "__main__":
    main()