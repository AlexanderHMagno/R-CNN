import os
import json
import xml.etree.ElementTree as ET

# Paths (Modify These)
ANNOTATIONS_DIR = "datasets/annotations"  # Change to your XML annotations folder
OUTPUT_JSON = "datasets/annotations.json"  # Where to save the COCO JSON

# COCO JSON Format
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object"}]  # Only one class
}

annotation_id = 0

# Process Each XML File
for xml_file in os.listdir(ANNOTATIONS_DIR):
    if not xml_file.endswith(".xml"):
        continue

    
    try:
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()
    except ET.ParseError:
        print(f"Skipping file due to parsing error: {xml_file}")
        continue
    
    # Extract Image Info
    filename = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)
    image_id = len(coco_data["images"])
    
    coco_data["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })
    
    # Extract Objects
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        # Convert VOC bbox format (xmin, ymin, xmax, ymax) to COCO format (x, y, width, height)
        bbox_coco = [xmin, ymin, xmax, ymax]
        
        # Add Annotation
        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # Only one class
            "bbox": bbox_coco,
            "area": (bbox_coco[2] - bbox_coco[0]) * (bbox_coco[3] - bbox_coco[1]),
            "iscrowd": 0
        })
        annotation_id += 1

# Save to JSON File
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_data, f, indent=4)

print(f"COCO annotations saved to {OUTPUT_JSON}")
