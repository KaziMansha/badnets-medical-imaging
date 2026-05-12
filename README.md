# badnets-medical-imaging

# Instructions

## Setup

To download the data and get it set up/preprocessed:

1. Download MRI tumor dataset from [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download)
2. Extract dataset to `<repo root>/data/raw`, should have folders `<repo root>/data/raw/Training` and `<repo root>/data/raw/Testing` when done
3. Run `src/preprocessing.py`
4. Run through `01_preprocessing.ipynb`
   1. If you get Python package issues, install all relevant packages using `python -m pip install -r requirements.txt`

## Toy model

To set up and run the toy model, run through `02_toy_model.ipynb`.

# Model runs

Data to get results for, for each model:

- No change (control)
- JPEG compression (only to test since it's easy to apply, kind of like a slightly off control)
- Beam hardening
- Gaussian noise
- Gibbs ringing
- Motion ghosting
- Rician noise
- Combination of (beam hardening, ..., Rician noise), split corrupted data in 5 pieces

For each of these, we want to run on 0, 10, and 25 percent corrupted data to identify the impact of corruption of training data on test results.

The test results we will be running will be:

- Uncorrupted data
- 10% corrupted data
- 25% corrupted data
- 50% corrupted data
- 100% corrupted data

So in total there are [No change, JPEG compression, ..., Combination] x [0, 10, 25] x [Uncorrupted, ..., 100% corrupted] model runs that we need to do.

Once we have this data available, we can look into what worked the best.

# Idea/Thesis

Does adding corrupted images to the training dataset of a CNN improve its classification performance when presented with corrupted inputs? What is the tradeoff w.r.t. performance on uncorrupted inputs? Is there a sweet spot? 

Our goal is to test this specifically on medical data, such as data for tumor classification.

**Note:** Similar to [https://arxiv.org/abs/2406.17536](https://arxiv.org/abs/2406.17536). How can we differentiate?

## Corruption methods

Corruption methods:

- JPEG compression
- Artifacting

More realistic medical corruption methods:

#### 1. Hardware & Physics-Based Artifacts

*   **Gibbs Ringing (MRI):** Oscillations near sharp edges (like the skull) caused by truncated Fourier transforms.
    *   *Realism:* High. Occurs when high-frequency data is undersampled.
*   **Rician/Gaussian Noise (MRI/CT):** MRI noise is technically Rician, which is signal-dependent, unlike the simple additive Gaussian noise used in natural images.
*   **Beam Hardening (CT):** Dark streaks between dense objects (like bone or metal implants).
*   **Acoustic Shadowing (Ultrasound):** Dark regions behind dense structures (like gallstones) where sound waves cannot penetrate.

#### 2. Patient-Induced Artifacts

*   **Rigid Motion Blur:** Caused by the patient shifting in the scanner. Unlike "linear" motion blur, this often creates "ghosting" (repeated faint images).
*   **Respiratory/Cardiac Pulsation:** Rhythmic blurring or displacement specifically in chest or abdominal scans.

#### 3. Preparation & Lab Artifacts (Digital Pathology)

> **Not used in our experiments.** These are digital pathology artifacts, not MRI artifacts. Since our dataset consists of brain MRI scans, these corruptions are not clinically relevant and would dilute the analysis.

*   **Tissue Folds:** Physical folds in the tissue slice during slide preparation that double the thickness/opacity in a specific area.
*   **Stain Variation:** Differences in H&E color intensity due to different labs or chemical batches.
*   **Air Bubbles:** Circular distortions caused by air trapped under the coverslip.

### Strategic Impact on Performance

In medical AI, the "sweet spot" is much tighter. 
*   **The Risk:** Over-corrupting medical data can hide "micro-calcifications" or small "nodules" that are critical for diagnosis. 
*   **The Recommendation:** Use **Physically Accurate Augmentation**. Instead of random noise, use noise profiles specific to the scanner (e.g., Siemens vs. GE).
*   **The Trade-off:** Training on heavily blurred images might increase robustness to poor scans but will almost certainly decrease your model's **Sensitivity (Recall)** for early-stage disease detection, as the model learns to "ignore" the tiny details where disease first appears.