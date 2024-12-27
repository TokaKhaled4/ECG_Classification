# ECG Heartbeat Classification

## Overview

This project focuses on classifying ECG signals to differentiate between normal heartbeats and those indicative of premature ventricular contractions (PVCs). Precision is critical, as errors in ECG signal processing can have severe consequences for individuals' health. Our solution integrates signal processing, feature extraction, and machine learning to deliver a highly accurate classification system.

## Features

- **Pre-processing**: 
  - **Bandpass Filtering**: Removes noise outside the 0.5–40 Hz range.
  - **Normalization**: Scales data between -1 and 1 to ensure numerical stability and preserve signal characteristics.
- **Feature Extraction**: Utilizes Daubechies wavelet transform (db4) with three levels of decomposition.
- **Classification**: Implements a K-Nearest Neighbors (KNN) model, achieving 100% accuracy on the test set.

## Workflow

1. **Pre-processing**: Filters and normalizes raw ECG data.
2. **Feature Extraction**: Reduces data redundancy using wavelet transforms.
3. **Model Building**: Trains the KNN classifier to distinguish between normal and PVC signals.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, SciPy, scikit-learn, PyWavelets
- **Deployment**: Streamlit for a user-friendly interface

## Project Link

Access the live application here: [Project Link](#)  
