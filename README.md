# Pneumonia Detection with VGG16 and Grad-CAM: A Background-Aware Approach

This project explores the impact of background removal and data augmentation in detecting pneumonia from chest X-ray images using a VGG16-based CNN. It emphasizes explainability through Grad-CAM visualizations and performance analysis under different preprocessing scenarios.

---

## 🩺 Project Overview

- 📌 **Model**: VGG16 (pre-trained) fine-tuned for binary classification: Pneumonia vs. Normal
- 🌫️ **Experiment Focus**:
  - With vs. Without Background Removal
  - Before vs. After Data Augmentation
- 📊 **Evaluation**: Confusion matrix, classification report, Grad-CAM overlays
- ⚖️ **Class Imbalance**: Addressed using class weighting
- 🧠 **Explainability**: Visual inspection using Grad-CAM to interpret model attention

---

## 🧪 What This Project Contains

- Preprocessing code for background removal (OpenCV & segmentation-based)
- Data loaders and augmentation pipelines
- VGG16 training scripts using Keras
- Grad-CAM utility for explanation
- Detailed notebook comparisons for all 4 cases:
  1. With background, no augmentation
  2. With background, with augmentation
  3. Without background, no augmentation
  4. Without background, with augmentation ✅ (Best Results)

---

## 🚀 How to Run

1. Download the dataset from [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
2. Clone this repository:
   ```bash
   git clone https://github.com/viji1804/pneumonia-vgg16-gradcam.git
   cd pneumonia-vgg16-gradcam
   ```
Removing irrelevant background improves model focus and reduces noise.
Grad-CAM helps build trust in medical ML models through visual interpretation.
Comparing conditions (with/without BG, augmentation) reveals preprocessing’s real value.

Outputs:




![pic1](https://github.com/user-attachments/assets/76d4a4cf-5307-485e-ac7e-ea6cf323f24b)
![pic2](https://github.com/user-attachments/assets/86d782fc-4a2e-46ae-b95f-92f7caf63aab)
![pic3](https://github.com/user-attachments/assets/4a7ce274-a970-449d-a140-929ea225029f)
![pic4](https://github.com/user-attachments/assets/546794a5-68b3-40c9-8f64-e2d180f2e0c8)
![pic5](https://github.com/user-attachments/assets/a9dbaf06-a263-402e-8420-ee75f5f09900)
