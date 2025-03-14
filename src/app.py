import torch
import torchvision
import torchvision.transforms as T
import gradio as gr
from PIL import Image, ImageDraw
import torchvision.ops as ops
import numpy as np
import json
import os


# LOAD_MODEL_PATH = "models/lego_fasterrcnn.pth"
LOAD_MODEL_PATH = "models/faster_rcnn_custom.pth"

# Load trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

def mean_average_precision(predictions, ground_truths, iou_threshold=0.5):
    iou_scores = []
    for pred_box in predictions:
        best_iou = 0
        for gt_box in ground_truths:
            iou = compute_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        if best_iou >= iou_threshold:
            iou_scores.append(best_iou)
    return np.mean(iou_scores) if iou_scores else None

def predict(image, ground_truths_json=""):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    boxes = predictions[0]['boxes'].tolist()
    scores = predictions[0]['scores'].tolist()
    
    # Draw boxes on image
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        if score > 0.5:  # Confidence threshold
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{score:.2f}", fill="red")
    
    # Compute mAP if ground truths are provided
    mAP = None
    if ground_truths_json:
        try:
            ground_truths = json.loads(ground_truths_json)
            mAP = mean_average_precision(boxes, ground_truths, iou_threshold=0.5)
            # Draw ground truth boxes in a different color
            for gt_box in ground_truths:
                draw.rectangle(gt_box, outline="green", width=3)
                draw.text((gt_box[0], gt_box[1]), "GT", fill="green")
        except json.JSONDecodeError:
            print("⚠️ Invalid ground truth format. Expecting JSON array of bounding boxes.")
    
    # Filter boxes and scores based on confidence threshold
    filtered_boxes = [box for box, score in zip(boxes, scores) if score > 0.5]
    return image, filtered_boxes, mAP


def get_examples():
    # Load examples from JSON file
    with open("datasets/examples.json", "r") as f:
        examples_json = json.load(f)
    examples_with_annotations = examples_json["examples"]

    return examples_with_annotations

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(placeholder="Enter ground truth bounding boxes as JSON (optional)")],
    outputs=[gr.Image(type="pil", label="Detected LEGO pieces (Red predictions, green ground truth)"), gr.JSON(label="Predicted bounding boxes"), gr.Textbox(label="Mean Average Precision (mAP @ IoU 0.5)")],
    title="LEGO Piece Detector",
    examples=get_examples(),
    description="Upload an image to detect LEGO pieces using Faster R-CNN. Optionally, enter ground truth bounding boxes to compute mAP. If left empty, mAP will be null."
)

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
