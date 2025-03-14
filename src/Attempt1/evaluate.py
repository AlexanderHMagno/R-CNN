import torch
from utils import calculate_ap
from torchvision.transforms import functional as F

@torch.no_grad()
def calculate_map(model, data_loader, device, iou_threshold=0.5, num_classes=2):
    """Calculate Mean Average Precision (mAP) across all classes."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    for images, targets in data_loader:
        # Convert PIL images to tensors before sending to GPU
        images = [F.to_tensor(img).to(device) for img in images]
        predictions = model(images)
        
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    
    aps = []
    for class_id in range(1, num_classes):
        predictions_class = []
        targets_class = []
        
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            pred_labels = pred['labels'].cpu()
            target_boxes = target['boxes'].cpu()
            target_labels = target['labels'].cpu()
            
            pred_mask = pred_labels == class_id
            target_mask = target_labels == class_id
            
            predictions_class.append({'boxes': pred_boxes[pred_mask], 'scores': pred_scores[pred_mask]})
            targets_class.append({'boxes': target_boxes[target_mask]})
        
        ap = calculate_ap(predictions_class, targets_class, iou_threshold)
        aps.append(ap)
    
    return sum(aps) / len(aps)

@torch.no_grad()
def evaluate_model(model, val_loader, device):
    """Evaluate model performance using mAP."""
    mAP = calculate_map(model, val_loader, device)
    print(f"Validation mAP: {mAP:.4f}")
    return mAP
