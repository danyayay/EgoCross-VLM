import json
import pandas as pd

# Load all models
json_files = {
    'Qwen (Egocentric only, no CoT)': './logs/qwen_eval/Qwen3-VL-2B-Instruct/questionlast/20260310_115358_responses.json',
    'Qwen (Egocentric only, CoT)': './logs/qwen_cot_eval/Qwen3-VL-2B-Instruct/20260316_140731_responses_fixed.json',
    'Qwen (Egocentric + Gaze overlaid, CoT)': './logs/qwen_cot_eval/Qwen3-VL-2B-Instruct/20260317_110432_responses.json',
}

loaded_data = {}
for model_name, json_path in json_files.items():
    try:
        with open(json_path) as f:
            loaded_data[model_name] = json.load(f)
        print(f"Loaded {model_name}: {len(loaded_data[model_name])} samples")
    except FileNotFoundError:
        print(f"Warning: {json_path} not found")

# Test comparison logic
df_rows = []
for idx in range(5):  # Test first 5 videos
    no_cot_item = loaded_data['Qwen (Egocentric only, no CoT)'][idx]
    cot_only_item = loaded_data['Qwen (Egocentric only, CoT)'][idx]
    cot_gaze_item = loaded_data['Qwen (Egocentric + Gaze overlaid, CoT)'][idx]
    
    video_id = no_cot_item['video_id']
    answer = no_cot_item['answer']
    
    pred_no_cot = no_cot_item['pred_answer']
    pred_cot_only = cot_only_item['pred_answer']
    pred_cot_gaze = cot_gaze_item['pred_answer']
    
    no_cot_correct = pred_no_cot == answer
    cot_only_correct = pred_cot_only == answer
    cot_gaze_correct = pred_cot_gaze == answer
    
    print(f"\nVideo {video_id}:")
    print(f"  Answer: {answer}")
    print(f"  No CoT:     {pred_no_cot} {'✓' if no_cot_correct else '✗'}")
    print(f"  CoT Only:   {pred_cot_only} {'✓' if cot_only_correct else '✗'}")
    print(f"  CoT Gaze:   {pred_cot_gaze} {'✓' if cot_gaze_correct else '✗'}")
    
    # Check comparisons
    no_cot_better_than_cot_only = no_cot_correct and not cot_only_correct
    cot_only_better_than_no_cot = cot_only_correct and not no_cot_correct
    
    print(f"  Comparisons:")
    print(f"    No CoT > CoT Only: {no_cot_better_than_cot_only}")
    print(f"    CoT Only > No CoT: {cot_only_better_than_no_cot}")
