import torch
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("models/plane_detector.pth", map_location=device))
model.to(device)
model.eval()

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

def detect_planes(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    return prediction
