"""
Compare gaze analysis results across multiple evaluation runs or configurations.
Useful for comparing performance across different models or settings.
"""

import json
import sys
from pathlib import Path
from tabulate import tabulate


def load_analysis_json(filepath):
    """Load a gaze analysis JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_key_metrics(results):
    """Extract key metrics from a results dictionary."""
    return {
        'Total Samples': results['metadata']['total_samples'],
        'Sufficient Gaze (count)': results['sufficient_gaze']['count'],
        'Sufficient Gaze (acc)': results['sufficient_gaze']['accuracy'],
        'Sufficient Gaze (f1)': results['sufficient_gaze']['f1_score'],
        'Missing Gaze (count)': results['missing_gaze']['count'],
        'Missing Gaze (acc)': results['missing_gaze']['accuracy'],
        'Missing Gaze (f1)': results['missing_gaze']['f1_score'],
        'Overall Accuracy': results['overall']['accuracy'],
        'Overall F1': results['overall']['f1_score'],
    }


def create_comparison_table(results_dict):
    """Create a comparison table from multiple result dictionaries."""
    
    table_data = []
    headers = ['Metric'] + list(results_dict.keys())
    
    # Get all unique metrics from first result
    first_metrics = extract_key_metrics(list(results_dict.values())[0])
    
    for metric_name in first_metrics.keys():
        row = [metric_name]
        for config_name, config_results in results_dict.items():
            metrics = extract_key_metrics(config_results)
            value = metrics.get(metric_name, 'N/A')
            
            # Format numeric values
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        
        table_data.append(row)
    
    return tabulate(table_data, headers=headers, tablefmt='grid')


def compare_multiple_files(file_paths):
    """Compare multiple gaze analysis JSON files."""
    
    results = {}
    
    for fpath in file_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            print(f"Warning: File not found: {fpath}")
            continue
        
        config_name = fpath.stem.replace('gaze_analysis_', '').replace('_threshold', '')
        if not config_name:
            config_name = fpath.parent.name
        
        try:
            results[config_name] = load_analysis_json(fpath)
            print(f"✓ Loaded: {config_name}")
        except Exception as e:
            print(f"✗ Error loading {fpath}: {e}")
    
    if not results:
        print("No files loaded successfully")
        return
    
    print("\n" + "="*80)
    print("GAZE ANALYSIS COMPARISON")
    print("="*80)
    print(create_comparison_table(results))
    print("="*80 + "\n")
    
    # Additional insights
    print("\nKey Insights:")
    print("-" * 80)
    
    # Find best overall accuracy
    best_overall = max(results.items(), 
                      key=lambda x: x[1]['overall']['accuracy'])
    print(f"Best Overall Accuracy: {best_overall[0]} ({best_overall[1]['overall']['accuracy']:.4f})")
    
    # Find best with gaze
    best_with_gaze = max(results.items(),
                        key=lambda x: x[1]['sufficient_gaze']['accuracy'])
    print(f"Best Sufficient Gaze Accuracy: {best_with_gaze[0]} ({best_with_gaze[1]['sufficient_gaze']['accuracy']:.4f})")
    
    # Find best without gaze
    best_without_gaze = max(results.items(),
                           key=lambda x: x[1]['missing_gaze']['accuracy'])
    print(f"Best Missing Gaze Accuracy: {best_without_gaze[0]} ({best_without_gaze[1]['missing_gaze']['accuracy']:.4f})")
    
    print("-" * 80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_gaze_analysis.py <json_file1> [json_file2] [json_file3] ...")
        print("\nExample:")
        print("  python compare_gaze_analysis.py results1.json results2.json")
        print("\nOr compare all results in a directory:")
        print("  python compare_gaze_analysis.py logs/*/Qwen*/*threshold.json")
        sys.exit(1)
    
    json_files = sys.argv[1:]
    compare_multiple_files(json_files)
