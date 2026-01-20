# 🎨 CycleGAN Face–Sketch Synthesis

## 📌 Project Overview
This repository contains a professional implementation of a **Cycle-Consistent Adversarial Network (CycleGAN)** for high-fidelity, bidirectional image-to-image translation. The model is trained on the **Person Face Sketches dataset** to perform two core tasks:

- Converting real human face photographs into artistic sketches  
- Reconstructing realistic facial images from hand-drawn sketches  

The project provides an optimized **Google Colab training pipeline** and a **Flask-based web deployment** that enables real-time inference through a browser-based interface.

---

## 🚀 Features

### 🔄 Bidirectional Style Transfer
- Dual generators for **Photo → Sketch** and **Sketch → Photo**
- Cycle consistency preserves facial identity and structure

### ⚡ Performance Optimization
- Lightweight ResNet-based generator with **3 residual blocks**
- Image resolution fixed at **128 × 128**
- Optimized for fast training and inference on **Tesla T4 GPU**

### 🧠 Smart Inference Logic
- Automatically detects input domain (photo or sketch)
- Uses pixel intensity analysis to route input to the appropriate generator

### 🌐 Integrated Deployment
- Flask-based web interface for image uploads
- Public access enabled via **Ngrok tunneling**
- Real-time visualization of generated results

### 💾 Robust Checkpointing
- Automatic checkpoint saving to **Google Drive**
- Prevents training loss due to Colab session timeouts
- Stores epoch-wise, best, and final model weights

---

## 📂 Project Structure

```
CycleGAN-Face-Sketch-Synthesis/
├── q1_A-02_Gen-AI.ipynb # End-to-end training & deployment notebook
├── templates/
│ └── index.html # Flask frontend UI
├── static/
│ ├── uploads/ # Uploaded input images
│ └── results/ # Generated output images
├── model/ # Exported model files
└── drive/MyDrive/
└── Checkpoints_CycleGAN_Fast/
├── epoch_n.pth # Checkpoint after each epoch
├── best_model.pth # Best validation loss model
└── final_model.pth # Final inference model
```


---

## 🏗️ Technical Architecture

### Generator
- Encoder–Decoder **ResNet architecture**
- Instance Normalization for style consistency
- Tanh activation for output normalization

### Discriminator
- **PatchGAN discriminator**
- Operates on **70 × 70 image patches**
- Focuses on high-frequency facial and sketch details

---

## 📉 Loss Functions

- **Adversarial Loss:**  
  Mean Squared Error (LSGAN) for smoother gradients and training stability

- **Cycle Consistency Loss:**  
  L1 loss ensuring reversible image translation

- **Identity Loss:**  
  Preserves color and composition when input matches target domain

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Backend:** Flask
- **Networking:** PyNgrok
- **Hardware:** Google Colab (Tesla T4 GPU)
- **Utilities:** PIL (Pillow), tqdm

---

## 💻 Usage

### 1️⃣ Environment Setup

Install required dependencies:
pip install flask pyngrok pillow torch torchvision

### 2️⃣ Training

Open q1_A-02_Gen-AI.ipynb in Google Colab
Mount Google Drive for persistent checkpoint storage
Configure training parameters such as MAX_IMAGES and EPOCHS
Run all cells to train the model
Final weights are saved as final_model.pth

### 3️⃣ Web Interface (Inference)

Provide your NGROK_AUTH_TOKEN in the deployment cell
Run the Flask application
Access the generated Ngrok public URL
Upload a photo or sketch
The system automatically detects the input domain and generates the result

## 📊 Performance Metrics

| Metric                     | Value                     |
|---------------------------|---------------------------|
| Training Speed             | ~10.84 iterations/second |
| Average Generator Loss     | ~3.54 (by Epoch 10)      |
| Image Resolution           | 128 × 128 × 3            |


## 📜 License
This project is intended for academic, research, and educational use.
