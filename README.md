# Facial Expression Recognition

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.10%2B-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6%2B-blue?logo=opencv)
![License](https://img.shields.io/badge/license-MIT-green)
![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen)
![GitHub issues](https://img.shields.io/github/issues/esraessfib/facial-expression-recognition)


A deep learning model that classifies facial expressions into 7 categories: angry, disgust, fear, happy, neutral, sad, and surprise.

## Dataset

This project uses the FER2013 dataset from Kaggle. You need to download it separately and place it in the `data/fer2013/` directory.

### File Purpose Overview

- **data/fer2013/** - Contains the FER2013 dataset with images organized by emotion (angry, disgust, fear, happy, neutral, sad, surprise)
- **models/** - Stores trained model files for inference and deployment
- **notebooks/** - For experimental code, data exploration, and analysis
- **results/** - Contains output visualizations that demonstrate model performance
- **src/train.py** - Main script to prepare data, build CNN architecture, train model, and save results
- **src/predict.py** - Loads trained model and makes predictions on new images
- **src/utils.py** - Helper functions for data processing and model evaluation
- **src/visualize.py** - Functions to create training curves and prediction visualizations
- **requirements.txt** - Lists all Python dependencies needed to run the project
- **README.md** - Project documentation and instructions
- **.gitignore** - Specifies files to exclude from version control (large datasets, cache files, etc.)

## 🎯 Results

### Training Performance
![Training History](results/training_history.png)

### Model Accuracy
- Training Accuracy: XX%
- Validation Accuracy: XX%
- Test Accuracy: XX%

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)


## 🏗️ Model Architecture

The model uses a CNN architecture with:
- 4 convolutional layers with BatchNormalization and MaxPooling
- 2 fully connected layers
- Dropout for regularization
- Softmax output layer with 7 units (one for each emotion)

## 📊 Dataset

This project uses the FER2013 dataset from Kaggle. The dataset contains:
- 28,709 training images
- 3,589 test images
- 7 emotion categories: angry, disgust, fear, happy, neutral, sad, surprise

Download it from [here](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the `data/fer2013/` directory.


### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/facial-expression-recognition.git
cd facial-expression-recognition
