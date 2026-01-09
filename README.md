# Violence Detection in Videos using Deep Learning

An end-to-end deep learning system for detecting violent activity in videos using spatiotemporal modeling. The project focuses on sequence-based video understanding by combining CNN-based feature extraction with temporal modeling using Bi-directional LSTMs.

---

## Overview

Detecting violence in videos requires understanding **temporal context**, not just individual frames.  
This project builds a complete pipeline that processes raw videos, extracts frame sequences, and learns temporal patterns to classify videos as **Violence** or **Non-Violence**.

The emphasis is on:
- Robust video preprocessing
- Sequence modeling
- Practical inference (frame-level and video-level)
- Realistic evaluation

---

## Key Features

- End-to-end video classification pipeline
- Frame sequence extraction with fixed temporal windows
- Spatiotemporal modeling using CNN + Bi-LSTM
- Frame-by-frame and full-video prediction
- Robust handling of corrupted or unreadable frames
- Clear evaluation using accuracy, confusion matrix, and classification report

---

## Dataset

- **Dataset:** Real-Life Violence Situations Dataset (Kaggle)
- **Classes:**
  - `NonViolence`
  - `Violence`

### Data Preparation
- Videos are sampled into fixed-length sequences of frames
- Frames are resized and normalized
- Videos with insufficient or corrupted frames are safely skipped

---

## Model Architecture

The model captures both spatial and temporal information:

Video Frames
↓
MobileNetV2 (TimeDistributed)
↓
Bi-Directional LSTM
↓
Fully Connected Layers
↓
Softmax Classification


### Architecture Highlights
- MobileNetV2 used as a lightweight feature extractor
- TimeDistributed wrapper to process frame sequences
- Bi-LSTM to model temporal dependencies
- Fine-tuning applied to higher CNN layers

---

## Training Details

- Image size: `64 × 64`
- Sequence length: `16 frames`
- Optimizer: SGD
- Loss: Categorical Cross-Entropy
- Regularization: Dropout
- Callbacks:
  - Early Stopping
  - ReduceLROnPlateau

---

## Results

- **Test Accuracy:** 94%
- Balanced precision and recall across both classes
- Stable convergence after fine-tuning

### Evaluation Metrics
- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)

---

## Inference Capabilities

- **Video-level prediction:** Classifies an entire video
- **Frame-by-frame prediction:** Annotates frames with predicted class
- Supports exporting annotated output videos for visualization

---

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

---

## Project Structure

├── data/
│ ├── Violence/
│ └── NonViolence/
├── notebooks/
│ └── training_and_evaluation.ipynb
├── models/
│ └── mobilenet_bilstm.h5
├── utils/
│ └── video_processing.py
└── README.md


---

## Limitations & Learnings

- Performance depends heavily on frame sampling strategy
- Subtle violent actions are harder to detect than explicit ones
- Temporal modeling significantly outperforms frame-only approaches
- Robust preprocessing is critical when working with real-world video data

---

## Future Improvements

- Experiment with 3D CNNs or Transformer-based video models
- Improve handling of subtle and long-duration violence
- Integrate audio-based signals
- Optimize for real-time inference

---

## Notes

This project was built to explore **spatiotemporal deep learning**, not just achieve high accuracy. Emphasis was placed on understanding video pipelines, temporal dynamics, and model behavior under real-world constraints.

---

## Author

Pranav Raj  
Computer Science | Machine Learning | Deep Learning



