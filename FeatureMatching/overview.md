# FeatureMatching Directory Overview

This directory contains code and resources for the drone's computer vision and feature matching systems, particularly focusing on object detection and tracking.

- **`deep_template_matcher.py`**: Implementation of deep learning-based template matching algorithms.
- **`drone_sift_matcher.py`**: Implementation of traditional SIFT (Scale-Invariant Feature Transform) feature matching for drone imagery.
- **`siamese_finder.py`**: Script utilizing a Siamese Neural Network architecture to find and track specific reference patches (e.g., in real-time HD footage).
- **`train_siamese.py`**: Training script used to train the custom Siamese Neural Network model.
- **`siamese_mars_weights.pth`**: Trained PyTorch model weights for the Siamese network, trained on Martian surface datasets for robust feature extraction.
