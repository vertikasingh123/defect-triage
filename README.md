# Defect triage for factory parts: find it, then name it

**In one sentence:** a system that looks at a photo of a screw, decides whether it's defective, and if so, says *what kind* of defect it is, using only a handful of example photos per defect type.

---

## Why this is a real problem

On a production line, cameras photograph every part. You want two answers for each photo:

1. **Is this part bad?**
2. **If yes, what's wrong with it?** A scratch on the head and a damaged thread come from different machines, so knowing the type tells you where to look for the cause.

Question 1 is easy to get data for: a good factory produces thousands of good parts, so you have plenty of photos of "normal." Question 2 is hard: defects are rare, so you might have five photos of each defect type. Five is not enough to train a normal image classifier.

This project shows how to answer both questions anyway.

---

## How it works

Two stages, one after the other.

**Stage 1: "Is it bad?"** Trained only on photos of good screws. It learns what normal looks like, patch by patch, and flags anything that doesn't match. It never sees a defect during training. As a bonus, because it checks each small patch of the image, it also knows *where* the odd part is.

**Stage 2: "What's wrong?"** A classifier that names the defect type (five types for screws). It is trained on just 5 real photos per type. To make those 5 go further, we tested adding artificial examples:

- **Cut-paste:** take the real defect out of one of the 5 photos (using its outline) and paste it onto a photo of a good screw, in the matching spot. Now you have 40 fake-but-realistic examples per type instead of 5.
- **Cut-paste + diffusion touch-up:** same, but run Stable Diffusion over the pasted edge so it blends in. We wanted to know if this helps.

And one more trick that turned out to matter most: instead of showing stage 2 the whole screw, **show it only the small area stage 1 flagged**. Small defects are a few pixels on a big photo; zooming in on them helps a lot.

```
photo ──► Stage 1: bad or good?  ──► if bad: where? ──► crop there ──► Stage 2: which defect type?
          (trained on good only)      (free from stage 1)             (5 real examples + synthetics)
```

---

## What we found

Tested on the MVTec AD "screw" dataset, three runs with different random choices of the 5 training photos.

**Stage 1 works well.** It ranks defective vs good screws correctly 97.5% of the time (AUROC 0.975). Set to catch 97% of defects, it wrongly flags 12% of good parts. It puts the true defect inside its "look here" crop 85% of the time.

**Stage 2, accuracy at naming the defect type (5 types, so guessing = 20%):**

| what the classifier sees | 5 real photos only | + cut-paste synthetics | + synthetics with diffusion touch-up |
|---|---:|---:|---:|
| the whole screw | 47% | 57% | 60% |
| **the area stage 1 flagged** | **77%** | **81%** | 79% |
| the exact defect area (cheating, for reference) | 84% | 89% | 87% |

Four things this table says:

1. **Zooming in on the area stage 1 flagged is the biggest win: +29 points**, from 47% to 77%, with no extra data. Small defects were simply too small to see in the full picture.
2. **Cut-paste synthetics help every time.** In all three runs, both with and without zoom, adding them improved accuracy (by 2 to 16 points depending on the setup). The gains land on the two hardest defect types.
3. **The diffusion touch-up didn't help.** Sometimes +2, sometimes −1, never outside the noise. We report this as a null result rather than hide it. The reason is fairly clear: the touch-up only smooths the edge of the paste and leaves the defect itself untouched, and once you zoom in, the edge barely matters.
4. **The next bottleneck is stage 1's aim, not more data.** When the crop is placed perfectly (the "cheating" row), accuracy rises another 7–8 points, and the hardest type (thread_side) doubles. Improving where stage 1 points is the obvious follow-up.

Per defect type, with zoom and cut-paste synthetics: three of the five types are at 97–100%. The two thread types are at 41% and 61%; those are the smallest defects and the ones stage 1 most often mislocates.

Figures and the full auto-generated reports are in [`figures/`](figures/).

---

## Running it yourself

You need Python 3.10+, a GPU (Intel Arc, NVIDIA, or CPU-only if patient), and the MVTec AD dataset (free, needs registration: https://www.mvtec.com/company/research/datasets/mvtec-ad).

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu   # Intel Arc
# NVIDIA: --index-url https://download.pytorch.org/whl/cu124     CPU only: no --index-url
pip install -r requirements.txt
```

Put `mvtec_ad.tar.xz` in `data/mvtec_ad/`, then run the scripts in order from the project root:

| script | what it does | time |
|---|---|---|
| `1_setup_mvtec.py` | extracts and checks the dataset | 2 min |
| `2_make_splits_and_synthetic.py` | picks the 5 training photos per type, makes cut-paste synthetics | 2 min |
| `3_refine_with_diffusion.py --limit 10` | touch-up on 10 images so you can eyeball `results/diffusion_comparison/` first | 1 min |
| `3_refine_with_diffusion.py` | touch-up on all of them | ~10 min on GPU |
| `4_anomaly_stage.py` | trains and evaluates stage 1 | 5 min |
| `5_train_classifier.py --regime all --input crop` | trains stage 2 (zoomed). Also try `--input full` and `--input oracle` | 2 min |
| `6_evaluate.py --input crop` | writes `results/REPORT_crop.md` and the figures | seconds |
| `7_export_onnx.py --input crop` | exports the stage-2 classifier as one ONNX file | 1 min |

Then repeat with other random draws of the 5 training photos, so the numbers aren't a fluke:

```powershell
.\run_seed.ps1 1
.\run_seed.ps1 2
```

Script 6 averages over every seed it finds. `config.yaml` has every knob, with comments.

---

## Things to know before quoting these numbers

- One dataset category, 94 test images per run: the margin of error on any accuracy is about ±5 points. That's why everything is run three times.
- Stage 1 picks the single most suspicious patch to crop around. Smoothing that decision, or checking a couple of candidate spots, would likely help the two weak defect types. Not done here.
- MVTec AD is licensed for non-commercial use. The code is reusable anywhere; models and synthetic images built from MVTec are not for commercial use.
- The ONNX export covers stage 2 only. Stage 1 still runs in PyTorch.

---

## Under the hood 

- **Stage 1** follows the PatchCore recipe: wide_resnet50_2 features from layers 2 and 3 at 320 px, a memory bank of 20,000 patches chosen by greedy k-center coreset from all good training images, image score = largest distance from any patch to its nearest bank patch. The crop centre is the patch with that largest distance.
- **Stage 2** uses frozen ImageNet ResNet50 features (2048-d, global pooled, 224 px input) with a StandardScaler and multinomial logistic regression. Every training image gets 8 flip/rotate variants. The three regimes differ only in what synthetic images are added.
- **Cut-paste** finds each screw's position and orientation (Otsu mask, principal axis, head end = wider end) and maps the defect from the source screw's frame to the target screw's frame, with small jitter. Pastes that don't land on the screw are rejected.
- **Diffusion touch-up** is Stable Diffusion 1.5 img2img at strength 0.3; the output is used only in a ring around the paste, and the defect pixels themselves are kept ("seam" mode). This was a deliberate choice after observing that letting the model repaint the defect tended to erase it.
- **ONNX export** folds the scaler and classifier into one linear layer and attaches it to the ResNet50 backbone; verified against the sklearn model to 4e-6.
