# 🧠 Script Identification for Indian Language Scene Text

This repository contains implementations of different deep learning models for identifying scripts in Indian language text images. The project explores multiple approaches to solve script classification using modern machine learning techniques.

---

## 📌 Overview

This project includes three main approaches:

### 🔹 ResNet
A CNN-based model using Residual Networks for image classification.

### 🔹 ViT (Vision Transformer)
A transformer-based model that processes images as patches and performs classification.

### 🔹 ParseqViT
A hybrid model combining transformer-based architectures for better performance.

---

## 📂 Repository Structure

Script-Identification-DC_AY-25-26/
│
├── ParseqViT/        # Parseq + ViT model
├── ResNet/           # ResNet implementation
├── ViT/              # Vision Transformer model
├── README.md

Each folder contains:
- Training code  
- Model architecture  
- Evaluation scripts  
- requirements.txt  

---

## ⚙️ Installation

### Step 1: Clone the repository
git clone https://github.com/codebythanos/Script-Identification-DC_AY-25-26.git  
cd Script-Identification-DC_AY-25-26  

---

### Step 2: Install dependencies

Go inside the required model folder:

Example (ViT):
cd ViT  
pip install -r requirements.txt  

Similarly for:
- ResNet  
- ParseqViT  

---

## 🚀 Usage

Navigate to the model folder you want to use.

### ▶️ Train
python train.py  

### ▶️ Test
python test.py  

### ▶️ Inference
python inference.py  

(Note: filenames may vary slightly based on your scripts)

---

## 🧪 Dataset

The project uses a 12-class Indian script classification dataset.

Typical structure:

dataset/
├── train_1800/
├── test_478/

Each class is stored in separate folders.

---

## 🔧 Features

- Multiple models (CNN + Transformer)
- GPU support
- Mixed precision training
- Hyperparameter tuning
- Confusion matrix & classification report

---

## 📊 Evaluation Metrics

- Accuracy  
- Class-wise Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  

---

## ⚠️ Requirements

- Python 3.8+
- TensorFlow
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- keras-hub (for ViT)

---

## 💡 Notes

- GPU is recommended (especially for ViT)
- Works on:
  - Google Colab  
  - Kaggle  
  - Local GPU setup  

---

## 🙏 Acknowledgements

- Vision Transformer (ViT) research  
- ResNet architecture  
- TensorFlow ecosystem  

---

## 👨‍💻 Author

Sumanth2006 (codebythanos)

---

## ⭐ Support

If you found this project useful, give it a ⭐ on GitHub!
