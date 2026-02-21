# Intel-Image-Classification-CNN-

## Project Overview
The objective of this project is to build and train a Convolutional Neural Network (CNN) to classify natural scenes into six distinct categories: Buildings, Forest, Glacier, Mountain, Sea, or Street. This project demonstrates the full machine learning lifecycle, from exploratory data analysis (EDA) and robust preprocessing to model optimization and comparative evaluation.

## Dataset Insights:
The project utilizes the Intel Image Classification dataset, originally created by Intel for an image classification challenge.

* Total Training Samples About: 14,000 images.
* Total Test Samples About: 3,000 images.
* Image Specifications: 150x150 pixels, 3-channel RGB.
* Images Categories: buildings, forest, glacier, mountain, sea, street.

## Technical Methodology
1. Data Preprocessing & Augmentation:
To improve model generalization and prevent overfitting, I implemented a comprehensive augmentation pipeline using Keras ImageDataGenerator:
* Scaling: Pixel values normalized to [0, 1] via rescale=1./255.
* Augmentation Techniques: Horizontal flipping, 20° rotation range, and 0.2% shifts in width, height, shear, and zoom.

2. Model Architectures:
I explored three different architectural approaches to identify the most effective solution:

* Basic Model: A custom CNN designed for lightweight inference.
* Deep Model: An enhanced CNN with deeper feature extraction layers and batch normalization.
* ResNet Model: Leveraging Transfer Learning with a pre-trained ResNet50 backbone to exploit high-level features learned from ImageNet.

3. Optimization Strategy
* Optimizer: Adam.
* Loss Function: Categorical Crossentropy.
* Callbacks: EarlyStopping: Prevents overfitting by monitoring validation loss.
* ReduceLROnPlateau: Dynamically lowers the learning rate when training plateaus.
* odelCheckpoint: Automatically saves the best-performing model weights.

## Accuracy Results:
* Basic Model:  %84.7
* Deep Model:   %87
* ResNet Model: %93.4

## Tools & Libraries
* Deep Learning: TensorFlow, Keras
* Data Manipulation: NumPy, Pandas
* Visualization: Matplotlib, Seaborn
* Evaluation: Scikit-learn (Classification Report, Confusion Matrix, Accuracy metrics, etc.)
