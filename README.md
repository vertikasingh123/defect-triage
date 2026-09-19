# Two-stage defect triage with few-shot type classification

**The problem.** On a production line, a one-class anomaly detector can flag a bad part using only images of good parts. It cannot tell you *what kind* of defect it found, and root-cause analysis needs the type: a scratch on the head points at one station, a thread defect at another. Real examples of each defect type are rare by design (a well-run line produces few defects), so a type classifier has to learn from a handful of images.

**The question this project answers.** With `k` real images per defect type (default 5), does adding synthetic defects help the type classifier, and does refining those synthetics with Stable Diffusion help further?

**The system.**

```
image ──► Stage 1: is it defective?          ──► Stage 2: what kind of defect?
          PatchCore-style kNN on ResNet           frozen ResNet50 features +
          patch features, trained on               logistic regression,
          GOOD images only                         k real images per class
                                                   (+ synthetic, under test)
```

Stage 2 is trained three ways on the *same* `k` real images and the *same* flip/rotate augmentation. The regimes differ only in what is added:

| regime | added data |
|---|---|
| `real_only` | nothing |
| `real_plus_synthetic` | cut-paste synthetics: the real defect (from MVTec's ground-truth mask) pasted onto clean parts with jitter |
| `real_plus_refined` | the same synthetics passed through Stable Diffusion img2img at low strength, which keeps the defect in place and harmonises the paste boundary |

Everything is evaluated on real held-out defect images that no stage ever saw.

---

## Setup

Python 3.10+. Tested layout: run every command from the project root.

```
defect-triage/
├── config.yaml
├── requirements.txt
├── README.md
└── scripts/
    ├── common.py                        shared helpers (do not run)
    ├── 1_setup_mvtec.py
    ├── 2_make_splits_and_synthetic.py
    ├── 3_refine_with_diffusion.py
    ├── 4_anomaly_stage.py
    ├── 5_train_classifier.py
    ├── 6_evaluate.py
    └── 7_export_onnx.py
run_seed.ps1                         one extra seed, end to end
```

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

**Install PyTorch for your hardware first.**

| hardware | command |
|---|---|
| Intel Arc / Intel iGPU (Windows or Linux) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu` |
| NVIDIA | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| CPU only | `pip install torch torchvision` |

Then `pip install -r requirements.txt`.

Check the GPU is seen (Intel):

```
python -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available())"
```

`config.yaml` has `device: "auto"` everywhere, which resolves to `xpu` → `cuda` → `cpu`. Nothing else needs changing.

**MVTec AD.** Free but requires registration: https://www.mvtec.com/company/research/datasets/mvtec-ad. Download `mvtec_ad.tar.xz` to `data/mvtec_ad/` and run script 1 to extract and verify.

---

## Run

```
python scripts/1_setup_mvtec.py
python scripts/2_make_splits_and_synthetic.py
python scripts/3_refine_with_diffusion.py --limit 10      # tune first, see below
python scripts/3_refine_with_diffusion.py
python scripts/4_anomaly_stage.py
python scripts/5_train_classifier.py --regime all
python scripts/6_evaluate.py
python scripts/7_export_onnx.py
```

### Then: let stage 1 tell stage 2 where to look

Small defects nearly vanish when a 1024 px photo is resized to 224 px and pooled to one vector. Stage 1
already computes a distance per patch, so its most anomalous patch says *where* the defect is. With
`--input crop`, stage 2 classifies a square crop around that location instead of the whole image.
Training crops are centred on the known masks; test crops use only stage 1's output, so the result is
what the deployed system would get. `--input oracle` centres test crops on the ground truth instead and
gives the ceiling if localisation were perfect.

```powershell
python scripts/4_anomaly_stage.py                          # once; also saves models/stage1_bank.npz
python scripts/5_train_classifier.py --regime all --input crop
python scripts/5_train_classifier.py --regime all --input oracle
python scripts/6_evaluate.py --input crop                  # report includes a full-vs-crop-vs-oracle table
```

The report also prints how often stage 1 put the true defect centre inside the crop.

### Then run more seeds (do this before quoting any stage-2 number)

The seed decides *which* k real images per class are used for training, and with k = 5 that choice
moves accuracy by several points. Scripts 2, 3 and 5 accept `--seed N`; each seed gets its own split,
its own synthetic folders (`data/synthetic/<category>_sN/`) and its own result files
(`results/stage2_<regime>_sN.json`). Script 6 averages whatever seeds it finds and reports mean ± sd.

```powershell
.\run_seed.ps1 1
.\run_seed.ps1 2
```

`run_seed.ps1` trains all three input modes (full, crop, oracle) for that seed. Each seed costs one diffusion pass (~10 min on a decent GPU) plus a couple of minutes. Three seeds is
the minimum for a claim; five is comfortable.

Rough timings on an Intel Arc iGPU: script 2 a couple of minutes; script 3 about 10–20 s per image (200 images at default `per_class: 40` × 5 classes ≈ 40–60 min; the first run also downloads ~4 GB of weights and compiles kernels); scripts 4–7 a few minutes each. Feature vectors are cached in `data/feature_cache/`, so re-running 5 and 6 is fast.

### Tune the diffusion step before running it in full

`--limit 10` refines two images per class and writes before/after pairs to `results/diffusion_comparison/`. Look at them. You want the defect still clearly present, the paste edge gone, the part otherwise unchanged.

- defect being erased or blurred → lower `diffusion.strength` (try 0.25)
- nothing visibly changed → raise it (try 0.40)
- the part changes shape or extra objects appear → lower `guidance_scale`, or tighten `negative_prompt`

Then run without `--limit`. Already-refined files are skipped, so you can stop and resume.

---

## What you get

`results/REPORT.md` is the write-up; it is generated from the numbers, so it never claims more than they show. Figures:

- `stage1_roc.png` — stage 1 good-vs-defective ROC and the operating point chosen for 95 % defect recall
- `fig1_regime_comparison.png` — accuracy and macro-F1 per regime
- `fig2_per_class_recall.png` — which defect types benefit from synthetic / refined data
- `fig3_confusion_matrices.png` — one per regime
- `stage2_<best>.onnx` + model card — single graph, image in → class probabilities out, validated against the sklearn model on real images

### What to expect, honestly

Stage 1 follows the PatchCore recipe (wide_resnet50_2, greedy coreset, 320 px) and lands around 0.97–0.98 AUROC
on `screw`. For reference, the same code with a random 10 % patch subsample at 224 px got 0.82: the coreset and
resolution matter.

Stage 2 at k = 5 will be noisy. Any of these outcomes is a legitimate result if you report it as such:

- synthetics help and refinement helps more — the clean story
- synthetics help, refinement is neutral — cut-paste artefacts were not the bottleneck
- synthetics hurt — the paste edge became a shortcut feature; refinement should then recover some of the loss, and that recovery is itself the finding

Run 3+ seeds (`run_seed.ps1`) before quoting a number; the report prints the refined-minus-raw
difference per seed and says whether its sign is consistent. On `screw`, expect `scratch_head` vs `scratch_neck` to be confused more than the rest: they differ mainly by location, and global pooled features discard location. If you want a category where defect types differ by appearance instead, set `dataset.name` to `hazelnut` or `metal_nut`; the tarball already contains every category.

---

## What is deliberately simplified

- Stage 1 uses a random 200k-patch pre-filter before the greedy coreset when the pool is larger than that
- Frozen backbone + linear head in stage 2 (k = 5 per class is too few to fine-tune without overfitting)
- Cut-paste keeps the defect near its source location with ±8 % jitter; MVTec parts are roughly centred so this is usually plausible
- No pixel-level localisation metrics; the deployment question is image-level type

---

## Talking about it

The one-line version: *built a two-stage defect triage system where stage 1 flags bad parts from good images only, and stage 2 identifies the defect type from five real examples per class; tested whether mask-guided cut-paste synthetics and Stable Diffusion refinement close the few-shot gap, evaluated on held-out real defects, exported as one validated ONNX graph.*

What to be ready to explain: why stage 1 needs no defect data and stage 2 does; why every regime gets the same real images and augmentation; how the scaler and classifier fold into one linear layer for export; what the per-class recall plot says about which defect types the synthetic data actually helped.
