# Image Classification with CNNs on FashionMNIST

## Overview
This project presents a comparative study of **convolutional neural network (CNN) architectures** for **image classification** on the **FashionMNIST dataset**.  
The implementation is done in **PyTorch** and evaluates both **custom-designed CNNs** and **standard deep architectures** using **transfer learning**.

The goal is to analyze how architectural choices, data augmentation, and pretrained weights affect model performance.

---

## Models Implemented

The following CNN-based models are implemented and evaluated:

### Custom CNN
- **VGG-like CNN**
  - Stacked convolutional blocks (Conv + BatchNorm + ReLU)
  - Max pooling and adaptive pooling
  - Fully connected classifier with dropout
  - Evaluated **with and without data augmentation**

### Pretrained CNN
- **ResNet18 Variants**
  - Random initialization
  - ImageNet pretrained backbone (frozen)
  - ImageNet pretrained backbone (fine-tuned)

---

## Dataset
- **FashionMNIST**
- 10-class image classification task
- Grayscale clothing images (28×28)

Dataset split:
- Training set
- Validation (development) set
- Test set

A fixed random seed is used to ensure **reproducible train/validation splits**.

---

## Data Preprocessing

- Image resizing:
  - 28×28 for custom CNN
  - 224×224 for ResNet18
- Channel handling:
  - 1-channel grayscale for VGG-style CNN
  - 3-channel RGB conversion for ResNet
- Normalization:
  - MNIST statistics for custom CNN
  - ImageNet statistics for pretrained ResNet
- Optional data augmentation:
  - Random rotation
  - Random affine transformations

---

## Training Details

- Loss function: **Cross Entropy Loss**
- Optimizer: **Adam**
- Validation-based model selection (best epoch chosen using validation loss)
- Training and validation loss curves are visualized
- Metrics computed on the test set using the best validation model

---

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-score (Macro & Micro)
- Per-class classification report
- Comparative performance summary exported as CSV

---

## Experiments

The experiments are organized into groups:
1. **VGG-style CNN**
   - Without augmentation
   - With augmentation
2. **ResNet18 variants**
   - Random initialization
   - ImageNet pretrained (frozen backbone)
   - ImageNet pretrained (fine-tuned)

Results are compared across all configurations to highlight the impact of:
- Data augmentation
- Network depth
- Transfer learning

---

## Language and Libraries Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- scikit-learn
- Matplotlib

---

## Purpose
This project was developed for educational purposes to:
- Gain hands-on experience with CNN architectures
- Understand data preprocessing and augmentation techniques
- Explore transfer learning with pretrained models
- Perform structured experimental comparisons in deep learning
