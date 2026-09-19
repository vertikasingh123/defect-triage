# 🔩 Defect Triage for Industrial Screws

### Few-Shot Defect Classification with Patch-Based Anomaly Detection and Synthetic Data Augmentation

An industrial visual inspection pipeline that first detects whether a screw is defective and then identifies the **type of defect** using only five real training images per defect category.

The project combines **PatchCore-based anomaly detection, defect localization, few-shot classification, cut-paste augmentation, and diffusion-based image refinement** to address the challenges of detecting and classifying rare manufacturing defects.

---

## 📌 Overview

Automated visual inspection systems need to answer two critical questions:

1. **Is the part defective?** — Detect anomalies without requiring defective samples during training.
2. **What is the defect?** — Classify the detected defect into a specific category, even when only a handful of labeled examples are available.

While normal production images are abundant, collecting labeled examples of every possible defect is expensive and impractical. This project addresses this challenge through a two-stage inspection pipeline.

### Key Highlights

* 🔍 Unsupervised anomaly detection trained exclusively on defect-free screws.
* 🎯 Patch-level anomaly localization to identify suspicious regions.
* 🧠 Few-shot defect classification using frozen ResNet-50 features.
* 🧩 Cut-paste synthetic defect generation from only five real examples per class.
* 🎨 Stable Diffusion-based refinement of synthetic defect boundaries.
* 📊 Evaluation across three random few-shot training splits.
* ⚡ ONNX export for the defect classification model.

---

## 🏗️ System Architecture

The pipeline consists of two sequential stages.

```text
                         INPUT IMAGE
                              │
                              ▼
               ┌──────────────────────────┐
               │      STAGE 1             │
               │   PatchCore Anomaly      │
               │       Detection          │
               │                          │
               │ Trained on good screws   │
               └────────────┬─────────────┘
                            │
                    Defect detected?
                       ┌────┴────┐
                       │         │
                      No        Yes
                       │         │
                       ▼         ▼
                  GOOD PART   ANOMALY MAP
                                  │
                                  ▼
                         Suspicious Patch
                                  │
                                  ▼
                         Localized Crop
                                  │
                                  ▼
               ┌──────────────────────────┐
               │      STAGE 2             │
               │ Few-Shot Defect          │
               │ Classification          │
               │                          │
               │ Frozen ResNet-50        │
               │ + Logistic Regression   │
               └────────────┬─────────────┘
                            │
                            ▼
                      DEFECT TYPE
```

### Stage 1 — Anomaly Detection and Localization

Stage 1 follows the **PatchCore** approach to identify defective screws without seeing any defective images during training.

* Extracts intermediate features from a pretrained `wide_resnet50_2` backbone.
* Uses features from layers 2 and 3 at a resolution of 320 pixels.
* Builds a memory bank containing representative normal image patches.
* Selects 20,000 memory-bank features using greedy k-center coreset subsampling.
* Computes the anomaly score using the largest nearest-neighbor distance between a test patch and the memory bank.
* Uses the most anomalous patch to estimate the defect location.

The result is both:

* An image-level anomaly score indicating whether a screw is defective.
* A suspicious patch location used to create the input crop for Stage 2.

> **Important:** Stage 1 is trained exclusively on defect-free training images and does not require defect labels.

### Stage 2 — Few-Shot Defect Classification

Stage 2 identifies the specific defect category using a small number of labeled examples.

The classifier uses:

* A pretrained ImageNet **ResNet-50** backbone with frozen weights.
* Global average-pooled 2048-dimensional feature vectors.
* StandardScaler for feature normalization.
* Multinomial logistic regression for five-class defect classification.
* Eight flip/rotation-based augmentation variants per real training image (synthetic images are added as generated).

Three training regimes were evaluated:

| Regime               | Training data                                     |
| -------------------- | ------------------------------------------------- |
| Real only            | Five real images per defect type                  |
| Cut-paste            | Real images + synthetic cut-paste samples         |
| Diffusion refinement | Real images + diffusion-refined synthetic samples |

The classifier was evaluated using three different inputs:

* **Full image:** The entire screw image.
* **Anomaly crop:** The region localized by Stage 1.
* **Oracle crop:** The ground-truth defect region, used only as a reference upper-bound experiment.

---

## 🧪 Synthetic Data Generation

To overcome the scarcity of labeled defect images, the project explores two synthetic data augmentation strategies.

### 1. Cut-Paste Augmentation

Defect regions are extracted from real defective screws and transferred onto defect-free screws.

The process includes:

1. Detecting the screw's position and orientation using an Otsu-based segmentation mask.
2. Estimating the principal axis of the screw.
3. Identifying the head end using the wider end of the screw.
4. Transforming the defect from the source screw's coordinate system to the target screw's coordinate system.
5. Applying small spatial perturbations to improve variation.
6. Rejecting samples where the pasted defect falls outside the target screw.
7. Saving the resulting synthetic defect image and its corresponding label.

This process adds **40 synthetic examples per defect category** on top of the 5 real images per category.

### 2. Diffusion-Based Refinement

To investigate whether generative models can improve synthetic image realism, Stable Diffusion 1.5 is used to refine the boundaries of pasted defects.

Configuration:

| Parameter       | Value                     |
| --------------- | ------------------------- |
| Model           | Stable Diffusion 1.5      |
| Task            | Image-to-image refinement |
| Strength        | 0.3                       |
| Refinement mode | Seam-only                 |
| Defect pixels   | Preserved                 |

The diffusion model is used only to refine a ring around the pasted region. The original defect pixels are retained to avoid accidentally removing or altering the defect.

This experiment investigates whether improving the visual blending of synthetic defects improves downstream classification performance.

---

## 📊 Experimental Results

Experiments were conducted on the **screw category of the MVTec AD dataset** using three independent random selections of five training images per defect category.

### Stage 1 — Anomaly Detection Performance

| Metric                                      |    Result |
| ------------------------------------------- | --------: |
| Image-level AUROC                           | **0.975** |
| True-positive rate / defect recall          |       97% |
| False-positive rate at that operating point |       12% |
| Defect localization coverage                |       85% |

The localization coverage measures how often the true defect falls within the crop generated from the most anomalous patch.

### Stage 2 — Defect Classification Accuracy

The task contains five defect categories, making random-guess accuracy 20%.

| Classifier input         | Real only | + Cut-paste | + Diffusion refinement |
| ------------------------ | --------: | ----------: | ---------------------: |
| Full screw image         |       47% |         57% |                    60% |
| **Stage 1 anomaly crop** |   **77%** |     **81%** |                **79%** |
| Oracle defect crop       |       84% |         89% |                    87% |

> Values are the mean over three random few-shot training splits (seeds 1, 2, 42); standard deviation across splits is 2–4 points. With 94 held-out test images per split, the sampling noise on any single accuracy is about ±5 points, which is why the experiments were repeated.

### 🔎 Key Findings

#### 1. Localization is critical for few-shot classification

Using the Stage 1 anomaly crop increased accuracy from **47% to 77%** in the real-only setting.

Small defects occupy only a few pixels in the full image. Cropping around the suspicious region provides the classifier with a more focused representation of the defect.

#### 2. Cut-paste augmentation improves classification

Cut-paste synthetic data improved accuracy across the evaluated input configurations.

The improvement was particularly relevant to the more challenging defect categories, including the thread-related defects.

#### 3. Diffusion refinement did not provide a consistent benefit

Diffusion-based seam refinement did not outperform cut-paste augmentation consistently.

The refinement modifies the paste boundary but leaves the actual defect unchanged. After localization and cropping, boundary realism may contribute less to classification than the underlying defect features.

This experiment is therefore reported as a **negative or null result**, rather than as evidence that diffusion refinement is universally ineffective.

#### 4. Localization is the next major bottleneck

Using the exact ground-truth defect region increased accuracy further, reaching 89% with cut-paste augmentation.

This suggests that improving Stage 1 localization could provide additional benefits, particularly for small thread-related defects.

### Per-Class Performance

With Stage 1 crops and cut-paste augmentation (per-class recall, i.e. the share of each type's test images labelled correctly):

* Three of the five defect categories reached approximately **97–100% recall**.
* The two thread-related categories reached **41% and 61% recall**.
* The thread-related categories were also among those most affected by localization errors.

These results highlight the difficulty of detecting and classifying small, visually similar defects.

---

## 📁 Project Structure

```text
.
├── scripts/
│   ├── common.py                        shared helpers (feature extraction, paths, image utilities)
│   ├── 1_setup_mvtec.py
│   ├── 2_make_splits_and_synthetic.py
│   ├── 3_refine_with_diffusion.py
│   ├── 4_anomaly_stage.py
│   ├── 5_train_classifier.py
│   ├── 6_evaluate.py
│   ├── 7_export_onnx.py
│   └── check_images.py                  removes truncated images left by interrupted runs
├── figures/                             plots and reports copied from a completed run
├── config.yaml
├── requirements.txt
├── run_seed.ps1
└── README.md

Generated locally when you run the pipeline (not in the repository):
├── data/                                dataset, splits, synthetic images, feature cache
├── models/                              stage-1 memory bank, stage-2 classifiers
└── results/                             reports, figures, ONNX export, diffusion before/after pairs
```

---

## ⚙️ Installation

### Requirements

* Python 3.10 or newer
* GPU recommended for accelerated execution

  * Intel Arc
  * NVIDIA CUDA-compatible GPU
  * CPU-only execution is supported but slower
* MVTec AD dataset

### 1. Clone the repository

```bash
git clone https://github.com/vertikasingh123/defect-triage.git
cd defect-triage
```

### 2. Create a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch

**Intel Arc GPU:**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

**NVIDIA GPU:**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**

```powershell
pip install torch torchvision
```

### 4. Install project dependencies

```powershell
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

This project uses the **screw category of the MVTec Anomaly Detection (MVTec AD) dataset**.

Download the dataset from the official website:

🔗 [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

The dataset requires registration and is licensed for non-commercial use.

Place the downloaded archive at:

```text
data/mvtec_ad/mvtec_ad.tar.xz
```

Then run the setup script to extract and validate the dataset.

---

## ▶️ Running the Pipeline

Run the scripts from the project root in the following order.

| Step | Script                                                    | Description                                |   Approx. time |
| ---: | --------------------------------------------------------- | ------------------------------------------ | -------------: |
|    1 | `scripts/1_setup_mvtec.py`                                | Extracts and validates the dataset         |          2 min |
|    2 | `scripts/2_make_splits_and_synthetic.py`                  | Creates few-shot splits and cut-paste data |          2 min |
|    3 | `scripts/3_refine_with_diffusion.py --limit 10`           | Refines 10 samples for visual inspection   |          1 min |
|    4 | `scripts/3_refine_with_diffusion.py`                      | Refines all synthetic samples              | ~10 min on GPU |
|    5 | `scripts/4_anomaly_stage.py`                              | Trains and evaluates Stage 1               |          5 min |
|    6 | `scripts/5_train_classifier.py --regime all --input crop` | Trains the Stage 2 classifier              |          2 min |
|    7 | `scripts/6_evaluate.py --input crop`                      | Generates evaluation reports and figures   |        Seconds |
|    8 | `scripts/7_export_onnx.py --input crop`                   | Exports the Stage 2 model to ONNX          |          1 min |

Each is invoked as `python <script>`, for example `python scripts/1_setup_mvtec.py`.

### Compare Different Classifier Inputs

```powershell
# Stage 2 using Stage 1 anomaly crops
python scripts/5_train_classifier.py --regime all --input crop
python scripts/6_evaluate.py --input crop

# Stage 2 using full screw images
python scripts/5_train_classifier.py --regime all --input full
python scripts/6_evaluate.py --input full

# Stage 2 using ground-truth defect crops
python scripts/5_train_classifier.py --regime all --input oracle
python scripts/6_evaluate.py --input oracle
```

### Evaluate Multiple Random Seeds

To repeat the experiments with different selections of the five real training images:

```powershell
.\run_seed.ps1 1
.\run_seed.ps1 2
```

Evaluation scripts aggregate the results across available seeds.

All major configuration parameters can be adjusted in `config.yaml`.

---

## 🧠 Technical Details

### PatchCore Configuration

| Parameter         | Value                                 |
| ----------------- | ------------------------------------- |
| Backbone          | `wide_resnet50_2`                     |
| Feature layers    | Layers 2 and 3                        |
| Input resolution  | 320 × 320                             |
| Training data     | Defect-free screws only               |
| Memory bank size  | 20,000 patches                        |
| Sampling          | Greedy k-center coreset               |
| Image-level score | Maximum patch-to-memory-bank distance |
| Localization      | Patch with the maximum anomaly score  |

### Stage 2 Configuration

| Parameter                   | Value                                            |
| --------------------------- | ------------------------------------------------ |
| Backbone                    | ImageNet-pretrained ResNet-50                    |
| Backbone weights            | Frozen                                           |
| Feature representation      | 2048-dimensional global pooled features          |
| Feature normalization       | StandardScaler                                   |
| Classifier                  | Multinomial logistic regression                  |
| Number of defect categories | 5                                                |
| Real images per category    | 5                                                |
| Augmentation                | 8 flip/rotation variants per real training image |

### ONNX Export

The Stage 2 classifier is exported as a single ONNX model.

The export process:

1. Extracts the ResNet-50 feature representation.
2. Folds the StandardScaler transformation and logistic regression parameters into a linear classification layer.
3. Attaches the classification layer to the ResNet-50 backbone.
4. Exports the combined model to ONNX.
5. Verifies the exported output against the original scikit-learn classifier.

The reported maximum numerical difference between the exported classifier and the original scikit-learn model was **4 × 10⁻⁶**.

> **Note:** The ONNX export covers Stage 2 only. Stage 1 continues to run in PyTorch.

---

## ⚠️ Limitations

* **Limited evaluation scope:** Experiments were conducted on the MVTec AD screw category, with 94 test images per run.
* **Few-shot variability:** Accuracy can vary depending on which five real training images are selected.
* **Localization errors:** Stage 1 selects only the single most anomalous patch, which can miss or misplace small defects.
* **Synthetic data limitations:** Cut-paste samples may not perfectly reproduce the physical appearance of real defects.
* **Diffusion refinement:** The evaluated seam-refinement approach did not consistently improve classification results.
* **Deployment scope:** Only the Stage 2 classifier is currently exported to ONNX.

---

## 🚀 Future Work

Potential improvements include:

* Using multiple candidate anomalous patches instead of only the top-scoring patch.
* Improving localization for small thread-related defects.
* Exploring multi-scale feature extraction.
* Evaluating more robust few-shot classification methods.
* Testing additional defect synthesis and augmentation techniques.
* Investigating end-to-end inference pipelines for production deployment.
* Extending ONNX export to the complete two-stage system.

---

## 📚 References

1. **PatchCore:** Roth et al. *Towards Total Recall in Industrial Anomaly Detection.* CVPR, 2022.
2. **MVTec AD:** Bergmann et al. *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR, 2019.
3. **Stable Diffusion:** Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR, 2022.
4. **ResNet:** He et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.

---

## 📄 Dataset Usage

The MVTec AD dataset is subject to its own license and usage restrictions. The dataset is intended for non-commercial research use, and models or synthetic images derived from it may be subject to applicable dataset terms.

Refer to the official [MVTec AD website](https://www.mvtec.com/company/research/datasets/mvtec-ad) for the complete dataset licensing conditions.

---

## 👩‍💻 Project Summary

This project demonstrates a practical approach to industrial defect triage under severe data scarcity.

By combining **unsupervised anomaly localization with few-shot defect classification**, the system separates the tasks of finding a defect and identifying its type. The experiments show that directing the classifier toward the anomalous region can be more impactful than simply generating additional training images.

The results also identify a clear direction for future improvement: **better defect localization, particularly for small thread-related defects.**
