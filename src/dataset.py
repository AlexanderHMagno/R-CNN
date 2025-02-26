import os
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class PlaneDataset(Dataset):
    def __init__(self, images_folder, annotations_folder, transform=None):
        self.images_folder = images_folder
        self.annotations_folder = annotations_folder
        self.transform = transform or T.ToTensor()
        self.image_filenames = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_folder, img_filename)

        # Load and convert image
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Read bounding boxes from CSV
        annotation_file = os.path.join(self.annotations_folder, img_filename.replace(".jpg", ".csv"))

        if not os.path.exists(annotation_file) or os.path.getsize(annotation_file) == 0:
            print(f"⚠️ Warning: Annotation file {annotation_file} is missing or empty!")
            return image, {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}

        try:
            bboxes_df = pd.read_csv(annotation_file, header=None, skiprows=1, sep=r"\s+")
            
            # Check if valid bounding boxes exist (at least 4 values per row)
            if bboxes_df.shape[1] != 4:
                print(f"⚠️ Warning: Invalid bounding boxes in {annotation_file}, skipping...")
                return image, {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}

            bboxes_df.columns = ["xmin", "ymin", "xmax", "ymax"]
            boxes = torch.tensor(bboxes_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        except Exception as e:
            print(f"❌ Error reading CSV {annotation_file}: {e}")
            return image, {"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)}

        target = {"boxes": boxes, "labels": labels}
        return image, target

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

dataset = PlaneDataset(images_folder="Images", annotations_folder="Annotations", transform=transform)
