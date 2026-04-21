

# OceanGuardian: Multimodal AI for Coral Bleaching Prediction

## Overview

**OceanGuardian** is an early-fusion multimodal deep learning framework designed to assess and predict coral bleaching events. By jointly reasoning across benthic imagery and real-time oceanographic sensor data, the system provides a holistic evaluation of marine ecosystem health.

This project emphasizes data integrity, specifically addressing multimodal challenges such as cross-modal data leakage, catastrophic forgetting, and class imbalance.

---

## System Architecture

The architecture employs an **Early Fusion** strategy to balance the influence of distinct data modalities:

* **Visual Branch:** Utilizes a pre-trained ResNet18 backbone. The final fully connected layer is constrained to 128 dimensions to prevent the visual modality from overpowering the tabular data.
* **Environmental Branch:** A Multi-Layer Perceptron (MLP) processes 6-dimensional tabular data (SST, pH, Latitude, Longitude, Month, Marine Heatwave) into a 64-dimensional feature vector, utilizing BatchNorm1d to stabilize distributions.
* **Fusion Module:** Features are concatenated (192 dimensions), synchronized via Batch Normalization, and regularized with Dropout (0.4) before passing through the final classification layer.

---

## Dataset & Preprocessing

The model is trained on a synthesized dataset combining:

1. **Visual Data:** 923 images of healthy and bleached corals.
2. **Tabular Data:** Oceanographic parameters including Sea Surface Temperature (SST) and pH levels.

**Data Leakage Prevention:**
To ensure strict evaluation, the pipeline separates Train and Validation splits independently for both image and tabular datasets *prior* to mapping. This strict stratification guarantees zero data overlap between training and validation phases.

---

## Performance Metrics

The model was optimized using **AdamW** and a **ReduceLROnPlateau** learning rate scheduler. Evaluation utilizes threshold calibration based on the **F1-Score** to maximize predictive balance.

* **Accuracy:** 85%
* **F1-Score:** 0.8541 (optimal threshold: 0.6655)
* **Precision / Recall (Bleached Class):** 0.90 / 0.81

**Environmental Feature Importance (Random Forest Evaluation):**

1. Longitude (0.2356)
2. Latitude (0.2182)
3. SST (0.2173)
4. pH Level (0.2102)

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/OceanGuardian.git
cd OceanGuardian
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation

Run the builder to process raw data and generate safe, non-leaking Train/Validation splits:

```bash
python src/data_builder.py
```

### 2. Training

Execute the training pipeline. The system will automatically apply class weights, L2 regularization, and save the best checkpoint (`best_model.pth`):

```bash
python train.py
```

### 3. Evaluation

Run the evaluation script to generate the classification report and confusion matrix:

```bash
python evaluate.py
```

### 4. Feature Analysis

Extract the importance of environmental parameters using a Random Forest ensemble:

```bash
python feature_analysis.py
```

### 5. Web Interface (SaaS Prototype)

Launch the interactive Gradio dashboard to perform real-time inference:

```bash
python app.py
```

---

## Directory Structure

```
OceanGuardian/
├── data/
│   ├── raw/
│   │   ├── healthy_corals/
│   │   ├── bleached_corals/
│   │   └── realistic_ocean_climate_dataset.csv
│   └── processed/
│       ├── best_model.pth
│       ├── train_split.csv
│       ├── val_split.csv
│       ├── confusion_matrix.png
│       └── feature_importance.png
├── src/
│   ├── data_builder.py
│   ├── dataset.py
│   └── model.py
├── app.py
├── train.py
├── evaluate.py
├── feature_analysis.py
├── requirements.txt
└── README.md
```

---

## Author

**Huy Nguyen**
Master of Science in Computer Science | Asian Institute of Technology (AIT)


