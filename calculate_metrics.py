#!/usr/bin/env python3
"""
Calculate evaluation metrics from fixed predictions JSON file.
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict

def calculate_metrics(json_file):
    """Calculate and print evaluation metrics."""
    
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract ground truth and predictions
    ground_truth = []
    predictions = []
    
    for entry in data:
        gt = entry.get('answer')
        pred = entry.get('pred_answer')
        
        if gt and pred:
            ground_truth.append(gt)
            predictions.append(pred)
    
    # Convert to numpy arrays
    gt_array = np.array(ground_truth)
    pred_array = np.array(predictions)
    
    # Calculate overall accuracy
    correct = np.sum(gt_array == pred_array)
    total = len(gt_array)
    accuracy = correct / total
    
    # Get unique classes
    classes = sorted(set(np.concatenate([gt_array, pred_array])))
    
    # Calculate confusion matrix manually
    cm = defaultdict(lambda: defaultdict(int))
    for gt, pred in zip(gt_array, pred_array):
        cm[gt][pred] += 1
    
    # Calculate per-class metrics
    precision = {}
    recall = {}
    f1 = {}
    support = {}
    
    for cls in classes:
        # Support is the number of samples in each class in ground truth
        support[cls] = np.sum(gt_array == cls)
        
        # True positives
        tp = cm[cls][cls]
        
        # False positives: predicted as cls but not actually cls
        fp = sum(cm[other][cls] for other in classes if other != cls)
        
        # False negatives: actually cls but predicted as something else
        fn = sum(cm[cls][other] for other in classes if other != cls)
        
        # Precision and Recall
        precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        if precision[cls] + recall[cls] > 0:
            f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
        else:
            f1[cls] = 0.0
    
    # Calculate macro F1
    macro_f1 = np.mean([f1[cls] for cls in classes])
    
    # Print results with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f"{timestamp} INFO Samples: {total}, Correct: {correct}/{total}, Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
    print(f"{timestamp} INFO ")
    print("Per-class metrics:")
    print(f"{timestamp} INFO class                          precision recall    f1        support")
    
    for cls in classes:
        print(f"{timestamp} INFO {cls:<30}{precision[cls]:.4f}    {recall[cls]:.4f}    {f1[cls]:.4f}        {support[cls]}")
    
    print(f"{timestamp} INFO ")
    print("Confusion matrix (rows=gt, cols=pred):")
    print(f"{timestamp} INFO {'':<36}", end="")
    for cls in classes:
        print(f"{cls:<12}", end="")
    print()
    
    for gt_cls in classes:
        print(f"{timestamp} INFO {gt_cls:<36}", end="")
        for pred_cls in classes:
            print(f"{cm[gt_cls][pred_cls]:<12}", end="")
        print()
    
    print(f"{timestamp} INFO ")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Incorrect predictions: {total - correct}")
    print(f"  Overall Accuracy: {accuracy:.4f}")
    print(f"  Macro F1 Score: {macro_f1:.4f}")
    print(f"\nClass Distribution:")
    for cls in classes:
        count = np.sum(gt_array == cls)
        print(f"  {cls}: {count} samples")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python calculate_metrics.py <path_to_responses.json>")
        sys.exit(1)
    json_file = sys.argv[1]

    calculate_metrics(json_file)
