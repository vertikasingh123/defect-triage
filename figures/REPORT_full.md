# Two-stage defect triage on MVTec AD / screw

**Question.** A one-class detector can flag a bad part with zero labeled defects, but it cannot say *what kind* of defect it is, and root-cause analysis needs the type. With only a handful of real examples per defect type, does adding cut-paste synthetic examples help a type classifier, and does refining those synthetics with Stable Diffusion help further?

## Stage 1 - is it defective? (good images only)

- image-level AUROC: **0.9751**
- at threshold 1.702 (chosen for TPR >= 0.95): 96.8% of held-out defects flagged, 12.2% of good parts falsely flagged (n = 94 defects, 41 good)
- memory bank: 20000 patches x 1536 dims (greedy coreset; wide_resnet50_2 layer2+layer3 at 320 px)

Classifier input: **whole image at 224 px**.

## Stage 2 - what kind of defect? (k = 5 real images per class, mean ± sd over 3 seeds [1, 2, 42])

| regime | extra training data | accuracy | macro-F1 | delta acc vs real-only |
|---|---|---:|---:|---:|
| real_only | - | 0.472 ± 0.025 | 0.470 ± 0.026 | +0.0000 |
| real_plus_synthetic | 200 cut-paste synthetic | 0.574 ± 0.038 | 0.561 ± 0.036 | +0.1028 |
| real_plus_refined | 200 diffusion-refined synthetic | 0.596 ± 0.028 | 0.585 ± 0.026 | +0.1241 |

Per-seed accuracy:

| seed | real_only | real_plus_synthetic | real_plus_refined |
|---|---:|---:|---:|
| 1 | 0.5000 | 0.5426 | 0.5638 |
| 2 | 0.4574 | 0.6170 | 0.6170 |
| 42 | 0.4574 | 0.5638 | 0.6064 |

Per-class recall on held-out real defects:

| class | real_only | real_plus_synthetic | real_plus_refined |
|---|---:|---:|---:|
| manipulated_front | 0.632 | 0.702 | 0.754 |
| scratch_head | 0.404 | 0.579 | 0.544 |
| scratch_neck | 0.583 | 0.800 | 0.800 |
| thread_side | 0.259 | 0.259 | 0.278 |
| thread_top | 0.463 | 0.500 | 0.574 |

## Reading the result

- Cut-paste synthetic data **helps**: +0.103 accuracy (positive in 3/3 seeds).
- Diffusion (seam) refinement gives **no measurable benefit** over raw cut-paste (+0.021; sign not consistent across seeds).
- Held-out set: 94 real defect images the classifier never saw in any form (one standard error on accuracy at this n is about ±0.052).
- Refined minus raw cut-paste, per seed [1, 2, 42]: +0.021, +0.000, +0.043 (mean +0.021). Sign is NOT consistent across seeds; treat as neutral.
- Raw cut-paste minus real-only, per seed [1, 2, 42]: +0.043, +0.160, +0.106 (mean +0.103). Consistent sign across seeds.

## Whole image vs. defect crop

| classifier input | real_only | real_plus_synthetic | real_plus_refined | thread_side recall (refined) |
|---|---:|---:|---:|---:|
| full | 0.472 | 0.574 | 0.596 | 0.278 |
| crop | 0.766 | 0.805 | 0.791 | 0.370 |
| oracle | 0.840 | 0.890 | 0.872 | 0.741 |

Reading: if 'crop' is well above 'full', small defects were a resolution problem, and stage 1's localisation is good enough to solve it in deployment. 'oracle' shows the ceiling if localisation were perfect.

## Method

- Stage 1: PatchCore recipe - wide_resnet50_2 layer2+layer3 patch descriptors at 320 px, greedy k-center coreset memory bank built from good images only, image score = max over patches of nearest-bank distance.
- Stage 2 features: frozen ImageNet resnet50, global pooled (2048-d) at 224 px.
- Stage 2: StandardScaler + multinomial LogisticRegression (C=1.0). All regimes see the same k real images with 8 flip/rotate variants; regimes differ only in added synthetic data.
- Synthetic: defect region cut from each real training image using MVTec's ground-truth mask, pasted onto a random clean image with +/-8 deg rotation, scale [0.9, 1.1], feathered edge.
- Refinement: Stable Diffusion img2img (strength 0.3, 25 steps, guidance 6.0); mode 'seam': only the ring around the paste is repainted; the defect's own pixels are kept. Everything outside a 24 px ring is the original pixel.

Figures: `fig1_regime_comparison.png`, `fig2_per_class_recall.png`, `fig3_confusion_matrices.png`, `stage1_roc.png`.