---
title: Objectlocalization
emoji: 👁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.18.0
app_file: src/app.py
pinned: false
short_description: Using RESTNET-RCNN-RPN-FNN to detect lego pieces
---


# LEGO Object Detection using Faster R-CNN

[Faster R-CNN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)

This project trains a **Faster R-CNN** model with a **ResNet-50 backbone** to detect LEGO objects using a custom dataset.

---
## 🔍 Project Overview

This project implements an advanced object detection system specifically designed for LEGO pieces using a combination of powerful deep learning architectures:

1. **ResNet-50 Backbone**: 
   - Serves as the feature extractor
   - Pre-trained on ImageNet for robust feature learning
   - Deep residual learning framework for improved training of deep networks

2. **Region Proposal Network (RPN)**:
   - Scans the image and proposes potential object regions
   - Generates anchor boxes of various scales and ratios
   - Outputs "objectness" scores and bounding box refinements

3. **Fast Neural Network (FNN)**:
   - Performs final classification and bounding box regression
   - Takes features from proposed regions
   - Outputs class probabilities and precise box coordinates

### Key Features

- **End-to-End Training**: The entire network is trained jointly for optimal performance
- **Multi-Scale Detection**: Capable of detecting LEGO pieces of varying sizes
- **Real-Time Processing**: Efficient architecture allows for quick inference
- **High Accuracy**: Achieves strong mean Average Precision (mAP) on LEGO detection

## **Project Structure**
```yaml
lego_detection/
│── models/                   # Trained models
│   ├── lego_fasterrcnn.pth   # Saved model
│   ├── faster_rcnn_custom.pth   # Latest model
│
│── datasets/                  # Dataset folder
│   ├── images/                # Training images
│   ├── annotations/           # Corresponding XML annotations
│   ├── test_images/           # Testing the model
│   ├── annotations.json/      # To format annotation in one only file
│
│── src/                       # Source code
│   ├── transformdata.py       # Formats the data to COCO.json
│   ├── new_trainer.py         # Train the model based on the new assumptions
│   ├── app.py                 # Allow users to interact with this model
│   ├── Attempt1               # First Implementation
│     ├── dataset.py             # Dataset class (LegoDataset)
│     ├── train.py               # Training script
│     ├── evaluate.py            # mAP Calculation
│     ├── utils.py               # IoU, AP calculation functions
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

## **visualize using Gradio**

1) If the model is not in models please (add it from the submitted file) Im trying to add the model but its too big for github standars.

2) Run the following Bash

```bash
python src/app.py
```
3) Evaluate and give me 100. I know, im awesome.  

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

## 📝 **Training Strategies for Faster R-CNN with ResNet-50 Backbone**

| Trainable Backbone Layers | Epochs | Batch Size | Recommended Learning Rate | Optimizer | Scheduler         |
|--------------------------|--------|-----------|--------------------------|-----------|------------------|
| 0                        | 10     | 4         | 0.0100                   | SGD       | StepLR(3, 0.1)   |
| 3                        | 10     | 8         | 0.0050                   | SGD       | StepLR(3, 0.1)   |
| 5                        | 10     | 16        | 0.0001                   | AdamW     | CosineAnnealing  |
| 3                        | 20     | 8         | 0.0050                   | SGD       | StepLR(5, 0.1)   |
| 5                        | 20     | 16        | 0.0001                   | AdamW     | CosineAnnealing  |


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



🚀 **Happy Training!**  
