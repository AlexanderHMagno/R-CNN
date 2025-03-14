import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import os
import json
from PIL import Image
from tqdm import tqdm  # Import tqdm for loading bar

# Paths (Modify These)
DATASET_DIR = "datasets/images"  # Folder containing images
ANNOTATIONS_FILE = "datasets/annotations.json"  # Path to COCO JSON

# Define Custom COCO Dataset Class (Without pycocotools)
class CocoDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.image_data = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = self.coco_data["annotations"]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        try:
            image_info = self.image_data[idx]
            image_path = os.path.join(self.root, image_info["file_name"])
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size  # Get image dimensions

            # Get Annotations
            annotations = [ann for ann in self.annotations if ann["image_id"] == image_info["id"]]
            boxes = []
            labels = []

            for ann in annotations:
                xmin, ymin, xmax, ymax = ann["bbox"]  # Now using [xmin, ymin, xmax, ymax]
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_width, xmax)
                ymax = min(img_height, ymax)
                
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(ann["category_id"])
                else:
                    print(f"⚠️ Skipping invalid bbox {ann['bbox']} in image {image_info['file_name']} (image_id: {image_info['id']})")

            if len(boxes) == 0:
                print(f"⚠️ Skipping entire image {image_info['file_name']} because no valid bounding boxes remain.")
                return None, None

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            target = {"boxes": boxes, "labels": labels}

            if self.transforms:
                image = self.transforms(image)

            return image, target
        except Exception as e:
            print(f"⚠️ Skipping image {image_info['file_name']} due to error: {e}")
            return None, None

# Define Image Transformations
transform = T.Compose([T.ToTensor()])

# Load Dataset
full_dataset = CocoDataset(root=DATASET_DIR, annotation_file=ANNOTATIONS_FILE, transforms=transform)
subset_size = min(10000, len(full_dataset))  # Limit dataset to 10,000 samples or less
subset_indices = list(range(subset_size))
dataset = Subset(full_dataset, subset_indices)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*[item for item in x if item[0] is not None])))

# Load Faster R-CNN Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Freeze Backbone Layers
for param in model.backbone.parameters():
    param.requires_grad = False

# Modify Classifier Head for Custom Classes
num_classes = 2  # One object class + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device("cpu")

# # Check for MPS Availability
# if torch.backends.mps.is_available():
#     print("✅ Using MPS (Apple Metal GPU)")
#     device = torch.device("mps")
# else:
#     print("⚠️ MPS not available, using CPU")
#     device = torch.device("cpu")

model.to(device)

# Training Setup
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 5

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    print(f"Epoch {epoch+1}/{num_epochs}...")
    
    for images, targets in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if any(len(t["boxes"]) == 0 for t in targets):
            print("⚠️ Skipping batch with no valid bounding boxes")
            continue

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save Trained Model
torch.save(model.state_dict(), "faster_rcnn_custom.pth")
print("Training Complete! Model saved as 'faster_rcnn_custom.pth'")
