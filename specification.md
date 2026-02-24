# Task: Can You Solve a Harder Inverse Problem? Ptychographic Phase Retrieval Under Sparse Scanning

**Objective:** Replicate a known baseline (PtychoNet), and develop an improved deep learning method for ptychographic phase retrieval. Your goal is to **maximize reduction in NRMSE** over the baseline.

## The Problem

Ptychography recovers an object's complex transmission function $T(r) = A(r) \cdot e^{i\phi(r)}$ from diffraction intensity measurements. A coherent probe $P$ scans across the object at $N$ overlapping positions. At each position $j$, the detector records:

$$I_j(q) = |\text{DFT}[P(r - r_j) \cdot T(r)]|^2$$

The detector records intensity only — **the phase is lost**.
Your job: given $\{I_j, r_j\}_{j=1}^N$ and the probe $P$, implement a method that recovers $T$ (both amplitude and phase).

The forward model is exact Fourier optics (~50 lines of FFT code, no approximations, no sim-to-real gap). What makes the inverse tractable is **overlap** between adjacent scan positions — but classical iterative methods like ePIE (https://ptycho.github.io/ptypy/) fail entirely below ~40% overlap. The sparse condition in this task has ~6% overlap.

## Task 1: Reimplement PtychoNet (Baseline)

PtychoNet (Guan & Tsai, BMVC 2019) is an encoder-decoder CNN for ptychographic phase retrieval. You must reimplement it as your starting point. The paper will be provided as an input to you in `/reimplement/original_paper.md`.

Your submission will be evaluated based on how well you replicate the results from the original paper.

## Task 2: Devise a Method that Outperforms the Baseline

Devise a method that outperforms PtychoNet. You may use any approach — architectural 
changes, loss function modifications, hybrid DL + iterative methods, or entirely new 
frameworks.

Your submission will be evaluated along two dimensions:
1. **Performance**: How much does your method reduce NRMSE relative to PtychoNet, 
   particularly under sparse scanning conditions? Failing to improve over the baseline 
   is the worst possible outcome.
2. **Approach**: Does your method reflect understanding of the ptychography problem? 
   A principled approach informed by the structure of ptychographic data or phase 
   retrieval physics will be valued more highly than arriving at the same NRMSE 
   through extensive hyperparameter tuning alone.

A thorough report with ablation studies, failure analysis, and clear reasoning will 
strengthen your evaluation.

## Rules & Constraints

### Evaluation Protocol

You must evaluate all methods under these exact conditions:
- **Ground truth objects:** Flickr30K images converted to complex objects (amplitude scaled to [0.5, 1.0], phase scaled to [−π/3, 0])
- **Probe:** Known Gaussian or Fresnel zone plate probe, 128×128 pixels
- **Scan pattern:** Regular grid, 6×6 diffraction pattern blocks
- **Three overlap conditions, all required:**

| Condition | Lateral Offset | Approx. Overlap | Difficulty |
|-----------|---------------|-----------------|------------|
| Dense     | 20 pixels     | ~69%            | Easy       |
| Moderate  | 40 pixels     | ~38%            | Medium     |
| Sparse    | 60 pixels     | ~6%             | Hard       |

- **Noise:** Noise-free (matching PtychoNet's original setup)
- **Test set:** At least 500 held-out Flickr30K images not seen during training
- **Metrics:** NRMSE for amplitude and phase separately, with global phase shift correction (find the constant $c$ minimizing $\|\hat{\phi} - \phi - c\|$). Report mean ± std over the test set.

### Instructions for Downloading the Dataset

Download the Flickr30K dataset from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("nlphuji/flickr30k", split="test")
train_data = dataset.select(range(28600))
test_data = dataset.select(range(28600, len(dataset)))
```

The dataset contains 31,783 images. Use the first 28,600 images for training 
and the remaining 3,183 for testing. Each image should be center-cropped, 
resized to 128×128 pixels, and converted to grayscale before use as ground-truth 
objects for simulating ptychographic measurements.

### No External Resources

- **Do not search the internet.** Do not use web search, browse documentation, or fetch external content beyond the dataset downloads above.
- **Do not read any local files or folders** outside of this experiment.
- **Create a single new folder** and keep all code, data, checkpoints, plots, and logs inside it.

### Compute Budget

- You are running in a resource-constrained environment (single machine, limited VRAM and time). Design experiments accordingly.
- **Babysit your experiments:** monitor training, catch divergence early, adjust hyperparameters, and iterate. Don't fire-and-forget.

### Autonomy

- Do not ask any questions. Make your own decisions and justify them in your write-up.

### No Reward Hacking

- The model must generalize to held-out test data it has never seen.
- You may not encode the answer in the input.
- You may not use an iterative solver or any non-learned method at inference time *unless* you explicitly frame it as a hybrid approach and report it separately from your pure DL results. (Hybrid approaches are allowed and encouraged — but you must also report the DL-only results.)

## What You Must Deliver

1. **PtychoNet baseline results.** Reimplement PtychoNet and report NRMSE (amplitude and phase) at all three overlap conditions. This is your starting point.
   - Add your code for reimplementation in `/reimplement`.

2. **Your improved method(s).** For each method you develop:
   - Architecture description (e.g., layers, dimensions, parameter count)
   - Loss function and training details (e.g., optimizer, LR schedule, batch size, training examples, training time, any curriculum)
   - Any additional attempt to build a novel method
   - NRMSE results at all three overlap conditions
   - Clear explanation of **what limitation of PtychoNet you identified** and **how your method addresses it**
   - Add your code in `/proposal`.

3. **Comparison table.** All methods, all conditions, one table:

| Method | Params | Amp NRMSE (20px) | Phase NRMSE (20px) | Amp NRMSE (40px) | Phase NRMSE (40px) | Amp NRMSE (60px) | Phase NRMSE (60px) |
|--------|--------|------------------|--------------------|------------------|--------------------|------------------|--------------------|
| PtychoNet (MSE) | ? | | | | | | |
| PtychoNet (DSE) | ? | | | | | | |
| Your method 1 | ? | | | | | | |
| Your method 2 | ? | | | | | | |

4. **Training curves.** Save plots of:
   - Training loss vs. step
   - Validation NRMSE vs. step (evaluated periodically at the sparse condition)
   - Any other diagnostics you find informative

5. **Failure analysis.** For your best method at the sparse condition (60px):
   - Show at least 10 failure cases (worst reconstructions) with predicted vs. ground truth amplitude and phase
   - Characterize what types of objects or structures your method struggles with

6. **A report** outlining everything you tried, including failed attempts, hyperparameter changes, and your reasoning at each decision point. Be transparent — the research process matters as much as the final result.
   - Save your report in `/proposal/report.md`.

Good luck.
