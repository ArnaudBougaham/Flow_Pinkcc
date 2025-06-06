# Medical Image Segmentation Lab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/medical-segmentation-lab/blob/main/teacher_version.ipynb)

## Overview
This lab focuses on medical image segmentation using CT scans. You will learn and implement best practices in medical image analysis, including:
- Exploratory Data Analysis (EDA)
- Data preprocessing and augmentation
- Train/Validation/Test splitting
- K-Fold Cross Validation
- Implementation of a U-Net architecture for 2D segmentation
- Model evaluation and visualization

## Dataset
The dataset consists of CT scans and their corresponding segmentation masks. The data is organized as follows:
- `Data/CT/`: Contains the CT scan images
- `Data/Segmentation/`: Contains the corresponding segmentation masks

## Lab Structure
The lab is designed to be completed in 3 hours. You will work with a Google Colab notebook that contains:
1. Data loading and exploration
2. Data preprocessing and augmentation
3. Model implementation
4. Training and evaluation
5. Results visualization

## Prerequisites
- Basic knowledge of Python
- Understanding of deep learning concepts
- Familiarity with PyTorch (basic level)

## Getting Started
1. Click the "Open in Colab" badge above to open the student notebook
2. Follow the instructions in the notebook
3. Complete the marked sections (look for `# TODO` comments)
4. Run the cells in sequence

## Learning Objectives
- Understand medical image data structure and preprocessing
- Implement proper data splitting and cross-validation
- Build and train a U-Net model for segmentation
- Evaluate model performance using appropriate metrics
- Visualize and interpret results

## Time Management
- Data exploration and preprocessing: 45 minutes
- Model implementation and training: 1.5 hours
- Evaluation and visualization: 45 minutes

## Note
This lab uses a subset of the full dataset to ensure it can be processed within Google Colab's memory constraints. The data has been preprocessed to work with 2D slices and includes class balancing techniques.

## Resources
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Medical Image Segmentation Tutorial](https://www.kaggle.com/code/iezepov/fast-ai-2018-lesson-3-notes) 