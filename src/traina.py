import os
import torch
import torchvision
import json
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from src.dataset import LegoDataset
from src.evaluate import calculate_map

# Load Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load Pretrained Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Freeze Backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Modify Predictor
num_classes = config["model"]["num_classes"]
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

# Load Dataset
image_dir = config["dataset"]["image_dir"]
annotation_dir = config["dataset"]["annotation_dir"]
dataset = LegoDataset(image_dir, annotation_dir)
train_size = int(config["dataset"]["train_split"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"], shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Logging Function (Writes logs efficiently to file)
log_file = "models/training_log.txt"
def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

# Training Function with tqdm progress bar
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Training Progress")):
        images = [F.to_tensor(img).to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)

# Train Model with Logging
num_epochs = config["model"]["epochs"]
os.makedirs("models", exist_ok=True)  # Ensure model directory exists

for epoch in range(num_epochs):
    log_message(f"Starting Epoch {epoch+1}/{num_epochs}")
    loss = train_one_epoch(model, optimizer, train_loader, device)
    log_message(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

    # Evaluate mAP after each epoch
    mAP = calculate_map(model, val_loader, device)
    log_message(f"Validation mAP: {mAP:.4f}")

    # Save log in JSON for visualization
    log_json = "models/training_log.json"
    if not os.path.exists(log_json):
        log_data = {"loss": [], "mAP": []}
    else:
        with open(log_json, "r") as f:
            log_data = json.load(f)

    log_data["loss"].append(loss)
    log_data["mAP"].append(mAP)

    with open(log_json, "w") as f:
        json.dump(log_data, f, indent=4)

# Save the trained model
torch.save(model.state_dict(), "models/lego_fasterrcnn.pth")
log_message("Model saved as models/lego_fasterrcnn.pth")
