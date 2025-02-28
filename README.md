---
title: Objectlocalization
emoji: 👁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.18.0
app_file: src/app.py
pinned: false
short_description: Using RCNN and Fully connected to detect Planes in objects
---


# LEGO Object Detection using Faster R-CNN

[Faster R-CNN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)

This project trains a **Faster R-CNN** model with a **ResNet-50 backbone** to detect LEGO objects using a custom dataset.

---

## **Project Structure**
```yaml
lego_detection/
│── models/                   # Trained models
│   ├── lego_fasterrcnn.pth   # Saved model
│
│── datasets/                  # Dataset folder
│   ├── images/                # Training images
│   ├── annotations/           # Corresponding XML annotations
│
│── src/                       # Source code
│   ├── dataset.py             # Dataset class (LegoDataset)
│   ├── train.py               # Training script
│   ├── evaluate.py            # mAP Calculation
│   ├── utils.py               # IoU, AP calculation functions
│
│── config.yaml                # Hyperparameters & settings
│── README.md                  # Project documentation
```

---

## ⚡ **Setup Instructions**
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Update Configuration**
Modify **`config.yaml`** to adjust **hyperparameters, dataset paths, and model settings**.

---

## 🚀 **Training the Model**
Run the following command to start training:
```bash
python src/train.py
```
This script will:
✅ Train Faster R-CNN with **LegoDataset**  
✅ Log training **loss & mAP**  
✅ Save the trained model in `models/lego_fasterrcnn.pth`

---

## 📊 **Monitoring Training Progress**
Use the Jupyter Notebook to **visualize loss & mAP over epochs**:
```bash
jupyter notebook notebooks/training_visualization.ipynb
```

---

## 🛠️ **Hyperparameters (`config.yaml`)**
Modify the **`config.yaml`** file to fine-tune the model:
```yaml
model:
  backbone: resnet50
  num_classes: 2
  pretrained: true
  learning_rate: 0.0001
  epochs: 5
  batch_size: 8
  optimizer: adam

dataset:
  image_dir: datasets/images
  annotation_dir: datasets/annotations
  train_split: 0.8
  val_split: 0.2

evaluation:
  iou_threshold: 0.5
```

---

## 📡 **Evaluating the Model**
Once training is complete, evaluate performance using:
```bash
python src/evaluate.py
```

---

## 💡 **Troubleshooting & Tips**
### ❓ **Training Takes Too Long?**
- Reduce `epochs` in `config.yaml`
- Use a **smaller dataset** for testing

### ❓ **mAP is too low?**
- Increase `epochs`
- Check dataset annotations
- Tune learning rate

---

## 🏆 **Contributors**
- 👤 **Alex** - Machine Learning Engineer

📧 **Contact**: [Your Email]  

🚀 **Happy Training!**  
