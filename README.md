# Machine Vision Character Classification using SVM (ME5405 - AY24/25)

This MATLAB-based project focuses on classical machine vision techniques and character classification using Support Vector Machines (SVM). The pipeline processes two provided images, extracts characters through segmentation, and classifies them based on a trained model. The project adheres to the constraints and guidelines set for the **ME5405 Computer Project** at NUS (Semester 1, AY24/25).

## Tools and Environment

- **Language**: MATLAB
- **Toolboxes Used**:
  - Image Processing Toolbox
  - Statistics and Machine Learning Toolbox

## 📷 Project Overview

### 🔹 Image 1: `Hello.jpg`
A basic preprocessing and filtering exercise:
- Grayscale conversion and contrast enhancement
- Image rotation correction using interpolation
- Smoothing via spatial filters (5×5 average and variable sizes)
- Frequency domain high-pass filtering
- Character segmentation

### 🔹 Image 2: `hello_world.jpg`
A more advanced task set:
- Sub-image extraction of the phrase `HELLO, WORLD`
- Thresholding to generate binary images
- Skeletonization (one-pixel thinning)
- Boundary detection (outline extraction)
- Character segmentation and labeling
- Character classification using **Support Vector Machines (SVM)**

## 🧠 Classification Task

Using the provided dataset `p_dataset_26.zip`, a supervised learning pipeline was implemented:

- **Training Data**: 75% of the dataset used for training the SVM model
- **Testing Data**: 25% of the dataset for model validation
- **Characters Recognized**: `H`, `E`, `L`, `O`, `W`, `R`, `D`
- **Preprocessing**: Image resizing and padding applied for consistency
- **Classifier**: MATLAB's `fitcsvm` used with hyperparameter experimentation (kernel functions, standardization, etc.)

**Bonus**: The classifier was also tested on Image 1 (`HELLO!`) to evaluate generalizability.

## Key Learnings & Observations

- Interpolation methods (nearest-neighbor, bilinear, bicubic) yield different smoothing/aliasing effects when correcting character slant.
- Spatial domain filtering (5×5) helps with noise reduction, while frequency-based high-pass filtering enhances edges.
- Character segmentation quality greatly affects downstream classification accuracy.
- SVM performance is sensitive to feature scaling, input size normalization, and kernel choice.




