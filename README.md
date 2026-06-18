# Automated Quality Control with Transfer Learning
### Manufacturing Defect Detection — Discovery-to-Action (DTA) Strategy

A transfer-learning pipeline that adapts a pre-trained ResNet50 CNN to classify
manufactured parts as **good** or **defective**, and translates the model's
predictions into concrete robotic-arm automation logic for a factory QC line.

---

## Project Structure

```
.
├── transfer_learning_quality_control.ipynb   # Main notebook (run this)
├── generate_synthetic_data.py                # Standalone version of the data generator
├── README.md
└── dataset/                                  # Created automatically when the notebook runs
    ├── train/
    │   ├── good/
    │   └── defective/
    └── test/
        ├── good/
        └── defective/
```

## How to Run

1. Open `transfer_learning_quality_control.ipynb` in **Google Colab**
   (`File → Upload notebook`, or open directly from this GitHub repo via
   `File → Open notebook → GitHub` and pasting the repo URL).
2. Set the runtime to GPU: `Runtime → Change runtime type → T4 GPU`.
3. `Run all`. Total runtime is roughly 5–10 minutes on a Colab GPU runtime
   (most of that is the 15-epoch training step).

No dataset download, API key, or external credentials are required — the
notebook generates its own dataset in Section 1.1 (see below).

It also runs in a local Jupyter environment with `tensorflow>=2.10` installed;
a GPU is recommended but not required (CPU training will simply be slower).

---

## 1. Dataset Preparation

This project uses a **synthetically generated** image dataset rather than a
downloaded one, by design: it guarantees the notebook is fully reproducible
for any reviewer with zero broken links, login walls, or multi-gigabyte
downloads, while still exercising every part of a real transfer-learning
pipeline.

**What the synthetic data looks like:** each image simulates a single
machined part (a ring/washer-style component) photographed against a noisy
gray conveyor-belt background.

- **`good`** — a clean part with normal machining texture lines.
- **`defective`** — the same base part with one or more of four overlaid
  defect types: **scratch**, **dent**, **crack**, or **stain**.

The generator (`generate_dataset()` in Section 1.1 of the notebook, also
available standalone as `generate_synthetic_data.py`) produces 300 `good` +
300 `defective` images for training and 75 + 75 for testing, saved into a
standard `dataset/train/{good,defective}/` and `dataset/test/{good,defective}/`
folder layout — the same layout `ImageDataGenerator.flow_from_directory`
expects for any real dataset.

### Preprocessing

- **Resize to 224×224×3** — the input resolution ResNet50 was trained on.
- **ResNet50-specific normalization** via
  `tensorflow.keras.applications.resnet50.preprocess_input`, which applies the
  exact channel-ordering and mean-subtraction the pre-trained backbone
  expects. This matters more than it might look: feeding a frozen backbone
  data normalized differently from its original training distribution
  causes its learned filters to respond to the wrong intensity ranges,
  degrading feature quality before the new head even gets a chance.

### Data Augmentation

Applied only to the **training** set (the test set is left undistorted, since
we want to evaluate on realistic, production-like images):

| Augmentation | Range | Rationale |
|---|---|---|
| Rotation | ±25° | Parts on a conveyor arrive at arbitrary angles |
| Width/height shift | ±10% | Camera framing/centering varies slightly |
| Horizontal & vertical flip | enabled | A defect's class doesn't depend on part orientation |
| Zoom | ±15% | Accounts for minor part-to-camera distance variation |
| Brightness | 0.85–1.15× | Simulates inconsistent factory floor lighting |

Aggressive augmentations (large shears, heavy color jitter) were deliberately
avoided, since they risk visually distorting small defects (e.g. a hairline
scratch) past the point of detectability, which would effectively corrupt
the label.

### Using a Real Dataset Instead

Drop any real dataset into the same `dataset/train|test/good|defective/`
folder structure and skip the `generate_dataset()` call — every other cell in
the notebook works unmodified.

---

## 2. Transfer Learning Workflow (The "Brain Swap")

1. **Import ResNet50** with `weights="imagenet"` and `include_top=False`,
   keeping only the convolutional feature-extraction backbone and discarding
   the original 1,000-class ImageNet output layer.
2. **Freeze the backbone** (`base_model.trainable = False`) so initial
   training only updates the new classification head. This preserves the
   generic visual features (edges, textures, gradients) ResNet50 already
   learned, avoids catastrophic forgetting from a randomly-initialized head's
   early large gradients, and keeps the trainable parameter count small
   relative to the dataset size — all of which protect against overfitting.
3. **Custom classification head:**
   `GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)`
4. **Compile** with binary cross-entropy loss and the Adam optimizer
   (`lr=1e-4`), and **train for 15 epochs** with `EarlyStopping` and
   `ReduceLROnPlateau` callbacks monitoring validation loss.

### Why GlobalAveragePooling2D instead of Flatten?

| | `Flatten()` | `GlobalAveragePooling2D()` |
|---|---|---|
| Output size | 7×7×2048 → 100,352-length vector | 2048-length vector (channel-wise average) |
| Params in next Dense(128) layer | ≈ 12.8M | ≈ 262K (about 49× fewer) |
| Overfitting risk | High, given a small dataset | Much lower |
| Spatial sensitivity | Preserves exact pixel-grid position of features | Discards position; keeps only "how strongly is this feature present, anywhere" |

A scratch's exact pixel coordinates don't matter for whole-part
classification — a scratch in the top-left of the frame is just as
defective as one in the bottom-right. `GlobalAveragePooling2D` encodes "is
this feature present somewhere in the image," which is the right inductive
bias for this task, while also cutting the parameter count dramatically —
the standard reason it's preferred over `Flatten()` in transfer-learning
classification heads built on frozen CNN backbones.

---

## 3. Performance Metrics

The notebook evaluates the trained model on the held-out test set using a
full **classification report** (precision, recall, F1) and a **confusion
matrix**, rather than accuracy alone — important for QC, since:

- **Recall (defective class)** measures how many real defects the model
  actually catches. Low recall means defective parts slip through to
  customers — typically the costlier failure mode.
- **Precision (defective class)** measures how many flagged parts were
  truly defective. Low precision means good parts get wrongly scrapped or
  sent for unnecessary manual review.

Run the notebook in Colab to generate the actual training curves,
classification report, and confusion matrix for this synthetic dataset —
these are saved as `training_curves.png` and `confusion_matrix.png` when the
notebook runs, and are also rendered inline.

---

## 4. Factory Decision Logic (Action Phase)

Predicted defect probabilities are converted into a three-band automation
policy for the robotic arm:

| Defect probability | Action | Robotic arm instruction |
|---|---|---|
| **≥ 0.85** | `REJECT` | Divert part to reject bin; log image + score |
| **0.50 – 0.849** | `HOLD_FOR_REVIEW` | Route part to manual inspection lane |
| **< 0.50** | `PASS` | Allow part to continue on the main conveyor |

The brief's specified 85% reject threshold is implemented as the
high-confidence cutoff, but a middle "hold for human review" band is added
rather than a single hard 85% cutoff — collapsing every part below 85%
straight to "pass" would let ambiguous, borderline cases through with zero
oversight, which is a worse policy than flagging them for a person to check.

**On threshold tuning:** these threshold values are a business policy
choice, not something learned by the model. They should ultimately be set
using the precision-recall tradeoff measured on real production data,
guided by the actual cost asymmetry on the line — a missed defect (false
negative) is usually far more expensive than an unnecessary manual review
(false positive), which argues for tuning the reject threshold against real
validation data rather than treating 85% as final.

---

## 5. Limitations & Next Steps

- **Synthetic data**: results here validate the *pipeline*, not real-world
  performance. Real factory defects (oxidation, weld spatter, glare, varied
  materials) are far more visually diverse than the four synthetic defect
  types used here.
- **Binary only**: the model doesn't classify defect *type*, which a real
  deployment likely needs for root-cause tracking.
- **No class imbalance handling**: the synthetic set is balanced 50/50; real
  defect rates are often under 5%, which would need class weighting or
  similar techniques to avoid the model collapsing to "always predict good."
- **Frozen backbone only**: a natural next step is fine-tuning the top 1–2
  ResNet50 blocks at a very low learning rate once the head has converged,
  letting the backbone specialize toward metal-defect textures specifically.

**For real deployment:** collect and label real line imagery, calibrate
thresholds against real precision-recall curves, fine-tune the backbone,
consider a defect-type sub-classification head, test latency on target edge
hardware, and run in shadow mode alongside existing QC before giving the
model reject authority.

---

## Author

Built as part of a Darey.io data science / ML curriculum project.
GitHub: [zaksz1991](https://github.com/zaksz1991)
