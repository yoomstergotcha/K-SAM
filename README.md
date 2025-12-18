# 👵 Identity-Preserving Korean Facial Age Transformation 

This repository contains the implementation of a three-stage framework for facial age progression and regression with identity preservation. Our work is inspired in part by Only a Matter of Style: Age Transformation Using a Style-Based Regression Model, which demonstrates that facial aging can be modeled as a continuous transformation in latent style space rather than a purely discrete attribute translation problem.

## Project Structure
- `final_demo.ipynb`: Standalone Colab demo notebook 
- `models/`: Generator, discriminator, and age estimator architectures
- `datasets/`: Paired age dataset loader
- `utils/`: Loss functions, evaluation, and visualization utilities

## Setup (Google Colab)
Open `final_demo.ipynb` and run all cells.  Upload your photo and adjust source / target age.
All required dependencies are installed automatically. 

## Method Overview
The pipeline consists of:
1. **Stage 1**: Age estimator pre-training (SSRNet)
2. **Stage 2**: Generator reconstruction pre-training
3. **Stage 3**: Age-conditioned fine-tuning with multi-loss supervision

## Repository Structure
```
K-Aging/
├── demo.ipynb                # Standalone Colab demo
├── models/
│   ├── generator.py          # ResNet-FiLM generators
│   ├── discriminator.py      # PatchGAN discriminator
│   ├── ssrnet.py             # Age estimator
│   └── encoders.py
├── datasets/
│   └── dataset.py            # PairedAgeDataset with landmarks
├── utils/
│   ├── losses.py
│   ├── geometry.py           # MediaPipe + warping
│   └── eval.py
├── training/
│   ├── train_stage2.py
│   └── train_stage3.py
├── checkpoints/              # (Tracked via Git LFS)
├── requirements.txt
└── README.md
```
