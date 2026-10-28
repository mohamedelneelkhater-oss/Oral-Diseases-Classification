# 🦷 Oral Diseases Image Classification

A deep learning project that classifies oral diseases from images, comparing a custom CNN
against fine-tuned pretrained models, with a Gradio app for the best one.

## Overview
- **Task:** Multi-class image classification of oral diseases ([N] classes)
- **Dataset:** [Dataset name / source on Kaggle]
- **Framework:** TensorFlow / Keras

## Pipeline
1. Data loading, exploration, and preprocessing
2. Data augmentation
3. Custom CNN built from scratch (baseline)
4. Transfer learning with 3 pretrained models: [Model 1], [Model 2], [Model 3]
5. Fine-tuning for every model
6. Automatic hyperparameter tuning
7. Callbacks: ReduceLROnPlateau, ModelCheckpoint (best weights), EarlyStopping
8. Evaluation: classification report, confusion matrix, per-class plots
9. Comparative analysis and discussion of results
10. Deployment of the best model with Gradio

## How to run
1. Open the notebook on Kaggle (or locally)
2. Install requirements: `pip install tensorflow gradio scikit-learn matplotlib`
3. Run all cells top to bottom

## Tech stack
Python · TensorFlow/Keras · scikit-learn · Matplotlib · Gradio
