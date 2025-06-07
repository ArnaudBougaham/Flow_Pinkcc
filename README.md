# Ovarian Cancer Segmentation Lab

## 👋 Welcome!
This lab introduces you to medical image segmentation using CT volumes and segmentation masks for ovarian cancer detection. You will work with real medical data, prepare volumetric images, build a U-Net model, and interpret the results.

## 📁 Project Structure

- `OvarianCancerSegmentation.ipynb`: Main lab notebook
- `Data_Subsample.zip`: Dataset (to be downloaded via the notebook)

## 🚀 Getting Started

1. Open the notebook in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArnaudBougaham/Flow_Pinkcc/blob/main/Ovarian_Segmentation_Teacher.ipynb)

2. Follow the instructions to download and extract the data
3. Run the cells in order, answer the questions, and interpret the results

## 🎯 Learning Objectives
- Understand volumetric medical data structure (NIfTI format)
- Perform exploratory data analysis on medical images
- Prepare and preprocess volumetric data for segmentation
- Implement and train a simple U-Net architecture
- Evaluate segmentation results and interpret medical outcomes

## 📊 Dataset Description
The dataset consists of:
- CT (Computed Tomography) volumes
- Segmentation masks with 3 classes:
  - Class 0: Background
  - Class 1: Primary ovarian cancer
  - Class 2: Metastasis

## 📋 Lab Structure
1. **Setup & Data Preparation**
   - Environment setup
   - Data download and extraction
   - Initial data exploration

2. **Exploratory Data Analysis**
   - Volume visualization
   - Class distribution analysis
   - Medical image characteristics

3. **Data Preprocessing**
   - Volume normalization
   - Data augmentation
   - Mask encoding

4. **Model Development**
   - U-Net architecture implementation
   - Loss function selection
   - Training pipeline setup

5. **Training & Evaluation**
   - Model training
   - Performance metrics
   - Results visualization

6. **Interpretation**
   - Medical significance
   - Model limitations
   - Potential improvements

## 💡 Tips for Success
- Pay attention to data normalization techniques
- Consider class imbalance in medical data
- Focus on both quantitative metrics and qualitative analysis
- Document your observations and medical interpretations 