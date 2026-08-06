"""
Analyze accuracy and F1 scores separately for eye gaze data categories:
1. Sufficient gaze data (>5 time-gaze pairs)
2. Missing gaze data (no valid gaze fixation)
3. Overall performance
"""

import json
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import sys


def extract_gaze_count(question_text):
    """
    Extract the number of gaze points from the question text.
    Returns:
        - Positive integer if gaze data exists
        - 0 if "No valid gaze fixation is detected"
    """
    if "No valid gaze fixation is detected" in question_text:
        return 0
    
    # Count occurrences of "Time" which indicates a gaze point
    # Pattern: "Time <value>s: Gaze(<x>, <y>)"
    gaze_pattern = r"Time\s+[-\d.]+s:\s+Gaze\(\d+,\s*\d+\)"
    matches = re.findall(gaze_pattern, question_text)
    return len(matches)


def normalize_answer(answer_text):
    """
    Normalize answer to a standard binary label.
    Handles: "cross"/"yield", "A"/"B", "cross (B)"/"yield (A)", etc.
    Returns: "cross" or "yield" (lowercase)
    """
    if not answer_text:
        return None
    
    answer_lower = answer_text.lower().strip()
    
    # Direct matches
    if "cross" in answer_lower:
        return "cross"
    elif "yield" in answer_lower:
        return "yield"
    
    return None


def get_uid(sample):
    """Try common keys to extract a unique id from a sample.
    Returns the uid value or None if not found.
    """
    if not isinstance(sample, dict):
        return None
    for key in ('uid', 'UID', 'Id', 'id', 'index', 'sample_id', 'video_id'):
        if key in sample:
            return sample[key]

    # check nested metadata
    for parent in ('meta', 'metadata'):
        val = sample.get(parent)
        if isinstance(val, dict):
            for key in ('uid', 'id', 'video_id'):
                if key in val:
                    return val[key]

    return None


def build_base_uid_map(base_json_path):
    """Load base JSON and return a dict mapping uid -> gaze_count.

    If a sample in the base file doesn't have a uid, it's skipped.
    """
    with open(base_json_path, 'r') as f:
        base_data = json.load(f)

    uid_map = {}
    for s in base_data:
        uid = get_uid(s)
        if uid is None:
            continue
        q = s.get('question', '')
        uid_map[uid] = extract_gaze_count(q)

    return uid_map


def analyze_gaze_performance(json_file_path, gaze_threshold=5, base_uid_gaze_map=None):
    """
    Analyze performance metrics for different gaze data categories.
    
    Args:
        json_file_path: Path to the JSON results file
        gaze_threshold: Minimum number of gaze points for "sufficient" category
    
    Returns:
        Dictionary with performance metrics for each category
    """
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize categories
    sufficient_gaze = {'true': [], 'pred': []}  # >= threshold gaze points
    missing_gaze = {'true': [], 'pred': []}      # 0 gaze points
    overall = {'true': [], 'pred': []}            # all samples
    
    categorized_count = {'sufficient': 0, 'missing': 0}
    skipped_count = 0
    # Track max gaze counts
    max_gaze_overall = 0
    max_gaze_examples = []  # store tuples (index, video_id, count)
    max_gaze_in_sufficient = 0
    
    # Process each sample
    for i, sample in enumerate(data):
        question = sample.get('question', '')
        true_answer = sample.get('answer', '')
        pred_answer = sample.get('pred_answer', '')
        
        # Normalize answers
        true_normalized = normalize_answer(true_answer)
        pred_normalized = normalize_answer(pred_answer)
        
        # Skip if we can't determine answers
        if not true_normalized or not pred_normalized:
            skipped_count += 1
            continue
        
        # Determine gaze count. If a base map was provided, prefer that
        # (this enables evaluating predictions in `json_file_path` using
        # the gaze labels from a separate `base_file`).
        gaze_count = None
        if base_uid_gaze_map is not None:
            uid = get_uid(sample)
            if uid is None:
                # can't match sample to base, skip it
                skipped_count += 1
                continue
            gaze_count = base_uid_gaze_map.get(uid)

        if gaze_count is None:
            gaze_count = extract_gaze_count(question)

        # Update max gaze overall
        if gaze_count > max_gaze_overall:
            max_gaze_overall = gaze_count
            max_gaze_examples = [(i, sample.get('video_id'), gaze_count)]
        elif gaze_count == max_gaze_overall:
            max_gaze_examples.append((i, sample.get('video_id'), gaze_count))
        
        # Add to overall
        overall['true'].append(true_normalized)
        overall['pred'].append(pred_normalized)
        
        # Categorize by gaze availability
        if gaze_count >= gaze_threshold:
            sufficient_gaze['true'].append(true_normalized)
            sufficient_gaze['pred'].append(pred_normalized)
            categorized_count['sufficient'] += 1
            # update max within sufficient category
            if gaze_count > max_gaze_in_sufficient:
                max_gaze_in_sufficient = gaze_count
        elif gaze_count == 0:
            missing_gaze['true'].append(true_normalized)
            missing_gaze['pred'].append(pred_normalized)
            categorized_count['missing'] += 1
    
    # Calculate metrics for each category
    results = {
        'metadata': {
            'total_samples': len(data),
            'processed_samples': len(overall['true']),
            'skipped_samples': skipped_count,
            'gaze_threshold': gaze_threshold,
            'max_gaze_overall': max_gaze_overall,
            'max_gaze_examples': [
                {
                    'index': idx,
                    'video_id': vid,
                    'gaze_count': cnt
                } for idx, vid, cnt in max_gaze_examples[:10]
            ],
            'max_gaze_in_sufficient': max_gaze_in_sufficient,
        },
        'sufficient_gaze': _calculate_metrics(
            sufficient_gaze['true'], 
            sufficient_gaze['pred'], 
            count=categorized_count['sufficient']
        ),
        'missing_gaze': _calculate_metrics(
            missing_gaze['true'], 
            missing_gaze['pred'], 
            count=categorized_count['missing']
        ),
        'overall': _calculate_metrics(
            overall['true'], 
            overall['pred'], 
            count=len(overall['true'])
        ),
    }
    
    return results


def _calculate_metrics(true_labels, pred_labels, count):
    """Calculate accuracy and F1 score for a set of labels."""
    
    if not true_labels:
        return {
            'count': 0,
            'accuracy': 0.0,
            'f1_score': 0.0,
            'f1_score_weighted': 0.0,
            'f1_macro': 0.0,
            'confusion_matrix': None,
            'class_breakdown': None,
        }
    
    acc = accuracy_score(true_labels, pred_labels)
    # Use 'cross' as the positive class for binary F1 score
    f1 = f1_score(true_labels, pred_labels, labels=['cross', 'yield'], average='binary', pos_label='cross', zero_division=0)
    f1_weighted = f1_score(true_labels, pred_labels, labels=['cross', 'yield'], average='weighted', zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, labels=['cross', 'yield'], average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=['cross', 'yield'])
    
    # Get detailed classification report
    class_report = classification_report(true_labels, pred_labels, labels=['cross', 'yield'], output_dict=True, zero_division=0)
    
    return {
        'count': count,
        'accuracy': float(acc),
        'f1_score': float(f1),
        'f1_score_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'confusion_matrix': cm.tolist(),
        'class_breakdown': class_report,
    }


def print_results(results):
    """Pretty print the analysis results."""
    
    print("\n" + "="*70)
    print("EYE GAZE PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Metadata
    meta = results['metadata']
    print(f"\nMetadata:")
    print(f"  Total samples: {meta['total_samples']}")
    print(f"  Processed samples: {meta['processed_samples']}")
    print(f"  Skipped samples: {meta['skipped_samples']}")
    print(f"  Max gaze points in any sample: {meta['max_gaze_overall']}")
    print(f"  Gaze threshold: {meta['gaze_threshold']}")
    
    # Sufficient gaze
    print(f"\n{'-'*70}")
    print(f"SUFFICIENT GAZE DATA (>= {meta['gaze_threshold']} gaze points)")
    print(f"{'-'*70}")
    _print_category_metrics(results['sufficient_gaze'])
    
    # Missing gaze
    print(f"\n{'-'*70}")
    print(f"MISSING GAZE DATA (No valid gaze fixation)")
    print(f"{'-'*70}")
    _print_category_metrics(results['missing_gaze'])
    
    # Overall
    print(f"\n{'-'*70}")
    print(f"OVERALL PERFORMANCE")
    print(f"{'-'*70}")
    _print_category_metrics(results['overall'])
    
    print(f"\n{'='*70}\n")


def _print_category_metrics(metrics):
    """Print metrics for a single category."""
    print(f"  Sample count:       {metrics['count']}")
    print(f"  Accuracy:           {metrics['accuracy']:.3f}")
    # print(f"  F1 Score (binary):  {metrics['f1_score']:.3f}")
    print(f"  F1 Score (macro):   {metrics['f1_macro']:.3f}")
    # print(f"  F1 Score (weighted):{metrics['f1_score_weighted']:.3f}")
    
    # if metrics['confusion_matrix']:
    #     cm = metrics['confusion_matrix']
    #     print(f"\n  Confusion Matrix (cross vs yield):")
    #     print(f"    True Cross:  [TN={cm[0][0]}, FP={cm[0][1]}]")
    #     print(f"    True Yield:  [FN={cm[1][0]}, TP={cm[1][1]}]")
        
    # if metrics['class_breakdown']:
    #     cb = metrics['class_breakdown']
    #     print(f"\n  Per-Class Metrics:")
    #     for label in ['cross', 'yield']:
    #         if label in cb:
    #             stats = cb[label]
    #             print(f"    {label.upper():6s}: precision={stats['precision']:.3f}, recall={stats['recall']:.3f}, f1={stats['f1-score']:.3f}, support={int(stats['support'])}")



def save_results_json(results, output_path):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_gaze_performance.py <json_file> [gaze_threshold] [base_file]")
        sys.exit(1)
    json_file = sys.argv[1]

    # Default threshold
    gaze_threshold = 1
    base_file = None

    # Accept flexible positional args after the json_file: any integer will be
    # interpreted as gaze_threshold, any other string treated as base_file path.
    for a in sys.argv[2:]:
        if a is None:
            continue
        try:
            gaze_threshold = int(a)
            continue
        except Exception:
            base_file = a

    print(f"Analyzing: {json_file}")
    print(f"Gaze threshold: {gaze_threshold}")
    if base_file:
        print(f"Using base file for gaze counts: {base_file}")

    # Build base map if provided
    base_uid_map = None
    if base_file:
        try:
            base_uid_map = build_base_uid_map(base_file)
            print(f"Loaded {len(base_uid_map)} uid entries from base file")
        except Exception as e:
            print(f"Warning: failed to load base file '{base_file}': {e}")
            base_uid_map = None

    # Run analysis
    results = analyze_gaze_performance(json_file, gaze_threshold=gaze_threshold, base_uid_gaze_map=base_uid_map)
    
    # Print results
    print_results(results)
    
    # Save results
    # output_file = Path(json_file).parent / f"gaze_analysis_{gaze_threshold}_threshold.json"
    # save_results_json(results, output_file)
