import torch
import torchvision.models as models
import torch.optim as optim
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from src.dataset import PlaneDataset, transform

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

# ✅ **Freeze Backbone (Feature Extractor)**
for param in model.backbone.parameters():
    param.requires_grad = False  # Prevents updating backbone layers

# Train only the detection head (Region Proposal + Classifier)
optimizer = optim.Adam(model.roi_heads.parameters(), lr=0.0001)

# Training loop
num_epochs = 5
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

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "models/frozen_plane_detector.pth")
