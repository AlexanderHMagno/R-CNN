import os
import torch
import torchvision
import gradio as gr
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Load Trained Model
def load_model(model_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # Background + 4 LEGO classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model("models/lego_fasterrcnn.pth")

def predict(image):
    image = Image.fromarray(image).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    results = []
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Confidence threshold
            results.append({
                "box": box.tolist(),
                "label": str(label),
                "score": float(score)
            })
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f"{label} ({score:.2f})", fill="red")
    
    return  image, results

def get_examples():
    return [os.path.join("datasets/test_images", f) for f in os.listdir("datasets/test_images")]

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="pil"), gr.JSON()],
    title="LEGO Detection with Faster R-CNN",
    description="Upload an image and the model will detect LEGO bricks with bounding boxes.",
    examples=get_examples()
)

demo.launch()
