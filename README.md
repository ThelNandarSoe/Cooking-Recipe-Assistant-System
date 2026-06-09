# Cooking-Recipe-Assistant-System
An innovative, real-time image processing and deep learning application designed to bridge the gap between food enthusiasts and culinary knowledge. The Cooking Recipe Assistant System (CRAS) automatically classifies food dishes from uploaded images and instantly generates their corresponding ingredients list and step-by-step cooking instructions.


# Cooking Recipe Assistant System 🍳🤖


An innovative, real-time image processing and deep learning application designed to bridge the gap between food enthusiasts and culinary knowledge. The Cooking Recipe Assistant System (CRAS) automatically classifies food dishes from uploaded images and instantly generates their corresponding ingredients list and step-by-step cooking instructions.

---

## 📌 Project Overview

Traditional recipe searching requires typing specific terms, which can be challenging when discovering unfamiliar global or local regional cuisines. This system leverages advanced Computer Vision (CV) and Convolutional Neural Networks (CNN) to make recipe discovery frictionless. 

By employing **Transfer Learning** via a pre-trained **ResNet50** architecture, the system is uniquely capable of recognizing a vast array of international foods as well as localized traditional cuisines.

### Key Core Objectives:
* **Intelligent Identification:** Accurately classify food dishes directly from user-uploaded images.
* **Dynamic Content Extraction:** Cross-reference classification maps to retrieve matching ingredient lists and clear, sequential cooking instructions.
* **Accessible UI:** Provide a simple, intuitive web app tailored for non-technical home cooks.

---

## 🚀 Key Features

* **Dual-Domain Recognition:** Trained to identify global food categories alongside specialized local cuisines.
* **On-the-Fly Image Preprocessing:** Handles automated resizing, noise reduction, and pixel normalization for optimized real-time evaluation.
* **Interactive Web Dashboard:** Built with Flask to enable instant image uploading and asynchronous results delivery.
* **Data Augmentation Processing:** Implements a robust image mutation pipeline (rotation, shift, zoom, shear, and brightness adjusting) inside training notebooks to drastically diminish validation overfitting.

---

## 🧠 Methodology & Architecture


The workflow progresses seamlessly from raw visual inputs to structured recipe responses:




The following figure shows the system flow of this project. 





<img width="496" height="732" alt="image" src="https://github.com/user-attachments/assets/806822aa-7f7b-4004-bc68-1596d8b30dfe" />







Figure 1.1 System flow for the project



### 1. Dataset Foundations
* **Global Base:** Utilizing the **Food101 Dataset** (101 food categories, 101,000 images).
* **Regional Fine-Tuning:** Augmented with a **Custom Burmese Traditional Food Dataset** comprising **20 additional categories** (e.g., *Lahpet*, *Mohinga*, *Shan Noodles*).
* **Recipe Database:** A structured backend relational mapping layout linking each distinct target category label directly to localized ingredient components and step-by-step instructions.

### 2. Model Development
* **Feature Extraction:** Leverages **ResNet50** initially trained on ImageNet weights to detect complex spatial hierarchies, boundaries, and intricate texture layouts.


🛠️ Technology Stack
Programming Language: Python

Deep Learning Framework: TensorFlow / Keras

Neural Network Backbones: ResNet50 Architecture

Web Deployment Server: Flask Web Framework

Data & Image Engineering libraries: OpenCV, NumPy, Pandas

