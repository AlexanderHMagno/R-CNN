import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import xml.etree.ElementTree as ET
import yaml 

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class LegoDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        # Limit dataset to 10000 images
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])[:config["model"]["image_sample_size"]]
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming 'lego' is class 1
        
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace(".jpg", ".xml"))
        
        image = Image.open(image_path).convert("RGB")
        boxes, labels = self.parse_annotation(annotation_path)
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target
