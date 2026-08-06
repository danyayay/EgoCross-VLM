#!/usr/bin/env python3
"""
Script to fix 'unknown' predictions by extracting answers from output_text.
Processes JSON files and attempts to extract the correct answer when pred_answer is "unknown".
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List


def extract_answer_from_text(output_text: str, options: Dict[str, str]) -> Optional[str]:
    """
    Extract the answer from output_text by looking for patterns.
    
    Tries multiple strategies:
    1. Look for "Final Answer:" patterns with letter/text
    2. Extract option letters (A, B, C, D, etc.)
    3. Look for actual answer text from options
    4. Look for action keywords in the last sentence
    
    Args:
        output_text: The output text to extract from
        options: Dictionary mapping letter to answer text (e.g., {"A": "yield", "B": "cross"})
    
    Returns:
        The extracted answer (value from options), or None if not found
    """
    if not output_text or not options:
        return None
    
    # Create reverse mapping: answer text -> letter
    reverse_options = {v.lower(): k for k, v in options.items()}
    
    # Strategy 1: Find pattern after "Final Answer" or "Answer"
    # Match patterns like "**Final Answer:** A" or "**Final Answer:** A)" or "**Final Answer:** yield"
    final_answer_match = re.search(
        r'(?:\*\*)?(?:Final\s+Answer|Answer)(?:\*\*)?[\s:]*\(?([A-Z])\)?(?:\s|$|[,.\)])' ,
        output_text,
        re.IGNORECASE
    )
    if final_answer_match:
        letter = final_answer_match.group(1).upper()
        if letter in options:
            return options[letter]
    
    # Strategy 2: Look for pattern at the very end - last standalone letter
    # Match the last occurrence of a single letter that maps to an option
    last_letter = None
    for match in re.finditer(r'\b([A-Z])\b', output_text):
        letter = match.group(1).upper()
        if letter in options:
            last_letter = letter
    
    if last_letter and last_letter in options:
        return options[last_letter]
    
    # Strategy 3: Look for the actual answer text from options (last occurrence wins)
    text_lower = output_text.lower()
    last_pos = -1
    best_answer = None
    for answer_text, letter in reverse_options.items():
        pos = text_lower.rfind(answer_text)
        if pos > last_pos:
            last_pos = pos
            best_answer = options[letter]
    
    if best_answer:
        return best_answer
    
    # Strategy 4: Try to find any option letters with more flexible pattern
    option_pattern = r'\b([A-Z])\)\s'
    matches = list(re.finditer(option_pattern, output_text, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        letter = last_match.group(1).upper()
        if letter in options:
            return options[letter]
    
    # Strategy 5: Look for action keywords in the last sentence (for variations like "stand still")
    # Extract last sentence
    sentences = re.split(r'[.!?]+', output_text.strip())
    if sentences:
        last_sentence = sentences[-1].lower().strip()
        
        # Look for action keywords
        for answer_text, letter in reverse_options.items():
            if answer_text in last_sentence:
                return options[letter]
    
    return None


def fix_unknown_predictions(json_file: Path, options: Dict[str, str] = None) -> tuple[int, int]:
    """
    Fix unknown predictions in a JSON file.
    
    Args:
        json_file: Path to the JSON file
        options: Optional pre-provided options dict
    
    Returns:
        Tuple of (total_fixed, total_unknown)
    """
    print(f"Processing {json_file}...")
    
    # Load JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"Error: Expected JSON array, got {type(data)}")
        return 0, 0
    
    fixed_count = 0
    unknown_count = 0
    details = []
    
    for i, entry in enumerate(data):
        if entry.get('pred_answer') == 'unknown':
            unknown_count += 1
            
            # Extract answer from output_text
            output_text = entry.get('output_text', '')
            entry_options = entry.get('options', options or {})
            
            extracted_answer = extract_answer_from_text(output_text, entry_options)
            
            if extracted_answer:
                old_pred = entry['pred_answer']
                entry['pred_answer'] = extracted_answer
                fixed_count += 1
                
                # Track for reporting
                details.append({
                    'index': i,
                    'video_id': entry.get('video_id', 'unknown'),
                    'old': old_pred,
                    'new': extracted_answer,
                    'ground_truth': entry.get('answer', 'unknown')
                })
                
                print(f"  ✓ Entry {i} ({entry.get('video_id')}): unknown -> {extracted_answer}")
            else:
                print(f"  ✗ Entry {i} ({entry.get('video_id')}): Could not extract answer from output_text")
    
    # Save the corrected JSON back to the same file
    if fixed_count > 0:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved corrected file to {json_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary for {json_file.name}:")
    print(f"  Total 'unknown' entries: {unknown_count}")
    print(f"  Successfully fixed: {fixed_count}")
    print(f"  Could not fix: {unknown_count - fixed_count}")
    print(f"{'='*60}\n")
    
    if details:
        print("Details of fixed entries:")
        for detail in details:
            match = "✓" if detail['new'] == detail['ground_truth'] else "✗"
            print(f"  {match} Index {detail['index']:4d} | {detail['video_id']:15s} | "
                  f"Extracted: {detail['new']:10s} | Ground truth: {detail['ground_truth']:10s}")
    
    return fixed_count, unknown_count


def process_directory(directory: Path, recursive: bool = True) -> None:
    """
    Process all JSON files in a directory.
    
    Args:
        directory: Directory to process
        recursive: Whether to search recursively
    """
    pattern = "**/*.json" if recursive else "*.json"
    json_files = list(directory.glob(pattern))
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return
    
    total_fixed = 0
    total_unknown = 0
    
    for json_file in json_files:
        fixed, unknown = fix_unknown_predictions(json_file)
        total_fixed += fixed
        total_unknown += unknown
    
    print(f"\n{'='*60}")
    print(f"OVERALL SUMMARY:")
    print(f"  Total files processed: {len(json_files)}")
    print(f"  Total 'unknown' entries fixed: {total_fixed}")
    print(f"  Total 'unknown' entries: {total_unknown}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_file() and path.suffix == '.json':
            fix_unknown_predictions(path)
        elif path.is_dir():
            recursive = len(sys.argv) > 2 and sys.argv[2] != '--no-recursive'
            process_directory(path, recursive=recursive)
        else:
            print(f"Error: {path} is not a valid JSON file or directory")
            sys.exit(1)
    else:
        print("Usage:")
        print("  python fix_unknown_predictions.py <json_file>")
        print("  python fix_unknown_predictions.py <directory> [--no-recursive]")
        print("\nExamples:")
        print("  python fix_unknown_predictions.py responses.json")
        print("  python fix_unknown_predictions.py ./logs")
        print("  python fix_unknown_predictions.py ./logs --no-recursive")
