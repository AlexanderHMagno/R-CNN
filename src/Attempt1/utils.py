import numpy as np

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def calculate_ap(predictions, targets, iou_threshold=0.5):
    """Calculate Average Precision (AP) for a single class."""
    all_pred_boxes = []
    all_pred_scores = []
    all_target_boxes = []
    
    for pred, target in zip(predictions, targets):
        all_pred_boxes.extend(pred['boxes'].tolist())
        all_pred_scores.extend(pred['scores'].tolist())
        all_target_boxes.extend(target['boxes'].tolist())
    
    sorted_indices = sorted(range(len(all_pred_scores)), key=lambda k: all_pred_scores[k], reverse=True)
    all_pred_boxes = [all_pred_boxes[i] for i in sorted_indices]
    all_pred_scores = [all_pred_scores[i] for i in sorted_indices]
    
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    
    for i, pred_box in enumerate(all_pred_boxes):
        best_iou = 0
        best_target_idx = -1
        
        for j, target_box in enumerate(all_target_boxes):
            iou = calculate_iou(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_target_idx = j
        
        if best_iou >= iou_threshold:
            tp[i] = 1
            all_target_boxes.pop(best_target_idx)
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(all_target_boxes) if len(all_target_boxes) > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    precisions = np.concatenate(([1], precisions))
    recalls = np.concatenate(([0], recalls))
    
    return np.trapz(precisions, recalls)
