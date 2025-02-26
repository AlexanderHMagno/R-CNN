import torch
import torchvision.models as models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import PlaneDataset, transform
from torchvision.ops import box_iou
import numpy as np

# Load dataset
dataset = PlaneDataset("Images", "Annotations", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Replace the classifier for detecting planes
num_classes = 2  # 1 for plane + 1 for background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Track statistics
train_losses = []
mAPs = []

# Function to compute mAP (mean Average Precision)
def compute_mAP(model, dataloader, device):
    model.eval()
    iou_threshold = 0.5
    all_precisions = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)
            
            for pred, target in zip(preds, targets):
                pred_boxes = pred["boxes"]
                pred_scores = pred["scores"]
                gt_boxes = target["boxes"]

                if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                correct = (ious.max(dim=1).values > iou_threshold).float()
                precision = correct.sum() / max(len(pred_boxes), 1)
                all_precisions.append(precision.item())

    return np.mean(all_precisions) if all_precisions else 0.0

# Training loop with statistics logging
num_epochs = 5
plt.ion()  # Turn on interactive mode for live plotting

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Compute and log statistics
    avg_loss = total_loss / len(dataloader)
    train_losses.append(avg_loss)
    mAP = compute_mAP(model, dataloader, device)
    mAPs.append(mAP)

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | mAP: {mAP:.4f}")

    # Live Plot Training Progress
    plt.figure(figsize=(10, 5))
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(mAPs, label="mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.title("Mean Average Precision")

    plt.pause(0.1)

# Save model
torch.save(model.state_dict(), "models/plane_detector.pth")
plt.ioff()  # Turn off interactive mode
plt.show() 
plt.savefig("plots/training_progress.png") # Show final plots
