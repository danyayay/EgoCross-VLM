import gradio as gr
from gradio.themes import Soft
import json
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Tuple
from itertools import combinations

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Define your models here with their properties and file paths
# Order matters - models will display in this order
MODELS_CONFIG = [
    # {
    #     'name': 'Qwen (Egocentric only, direct, qn3)',
    #     'path': './logs/qwen_qn3_direct_eval/Qwen3-VL-2B-Instruct/20260317_160241_responses.json',
    #     'tags': ['baseline'],  # Special tags for filtering/comparisons
    # },
    # {
    #     'name': 'Qwen (Egocentric + Gaze overlaid, CoT1, qn0)',
    #     'path': './logs/qwen_cot1_eval/Qwen3-VL-2B-Instruct/20260317_110432_responses.json', 
    #     'tags': ['with_gaze'],
    # },
    # {
    #     'name': 'Qwen (Egocentric only, CoT2, qn3)',
    #     'path': './logs/qwen_qn3_cot2_eval/Qwen3-VL-2B-Instruct/20260317_193128_responses.json',
    #     'tags': ['with_cot2'],
    # },
    # {
    #     'name': 'Qwen (Egocentric only, CoT1, qn3)',
    #     'path': './logs/qwen_qn3_cot1_eval/Qwen3-VL-2B-Instruct/20260317_161312_responses.json',
    #     'tags': ['with_cot1'],
    # },
    # {
    #     'name': 'Qwen (Egocentric only, CoT2, qn3)',
    #     'path': './logs/qwen_qn3_cot2_eval/Qwen3-VL-2B-Instruct/20260317_193128_responses.json',
    #     'tags': ['with_cot2'],
    # },
    # {
    #     'name': 'Qwen (Egocentric only, CoT4, qn3)',
    #     'path': './logs/qwen_qn3_cot4_eval/Qwen3-VL-2B-Instruct/20260317_224501_responses_fixed.json',
    #     'tags': ['with_cot4'],
    # },
    # {
    #     'name': 'Qwen (Egocentric + Fixation, CoT4, qn3)',
    #     'path': './logs/qwen_qn3_cot4_eval/Qwen3-VL-2B-Instruct/20260318_133146_responses_fixed.json',
    #     'tags': ['with_fixation'],
    # }
    # {
    #     'name': 'Qwen (Gaze overlaid, CoT4, qn3)',
    #     'path': './logs/qwen_eval_cot_wrongorder/qwen_qn3_cot4_eval/Qwen3-VL-2B-Instruct/20260319_054739_responses_fixed.json',
    #     'tags': ['with_overlay'],
    # },
    # {
    #     'name': 'Qwen (Egocentric + fixation, CoT5 new, qn3)',
    #     'path': './logs/qwen_qn3_cot5_new_eval/Qwen3-VL-2B-Instruct/20260322_232557_responses_fixed.json',
    #     'tags': ['with_cot5'],
    # },
    # {
    #     'name': 'Qwen (Egocentric + fixation, CoT6 new, qn3)',
    #     'path': './logs/qwen_qn3_cot6_new_eval/Qwen3-VL-2B-Instruct/20260322_235417_responses_fixed.json',
    #     'tags': ['with_cot6'],
    # }

    ### COT5 new with different input features (all qn3 for fair comparison)
    # {
    #     'name': 'CoT5 new, egocentric only',
    #     'path': './logs/qwen_new_qn3_cot5_eval/Qwen3-VL-2B-Instruct/20260323_210506_responses_fixed.json',
    #     'tags': ['baseline'],
    # },
    # {
    #     'name': 'CoT5 new, overlay',
    #     'path': './logs/qwen_new_qn3_cot5_eval/Qwen3-VL-2B-Instruct/20260323_230527_responses_fixed.json',
    #     'tags': ['with_overlay'],
    # },
    # {
    #     'name': 'CoT5 new, demographics',
    #     'path': './logs/qwen_new_qn3_cot5_eval/Qwen3-VL-2B-Instruct/20260324_042043_responses_fixed.json',
    #     'tags': ['with_demographics'],
    # }

    ### COT1 new with different input features (all qn3 for fair comparison), the reasoning here makes more sense.
    {
        'name': 'Egocentric only, CoT (simple)',
        'path': './logs/qwen_new_qn3_cot1_eval/Qwen3-VL-2B-Instruct/20260323_145938_responses_fixed.json',
        'tags': ['baseline'],
    },
    {
        'name': 'Egocentric with gaze overlay, CoT (simple)',
        'path': './logs/qwen_new_qn3_cot1_eval/Qwen3-VL-2B-Instruct/20260323_170806_responses_fixed.json',
        'tags': ['with_overlay'],
    },
    {
        'name': 'Egocentric + Demographics, CoT (simple)',
        'path': './logs/qwen_new_qn3_cot1_eval/Qwen3-VL-2B-Instruct/20260325_141427_responses_fixed.json',
        'tags': ['with_demographics'],
    }
    # {
    #     'name': 'Egocentric + demographics setup, CoT',
    #     'path': './logs/qwen_new_qn3_cot1_eval/Qwen3-VL-2B-Instruct/20260323_162236_responses_fixed.json',
    #     'tags': ['with_demographics'],
    # }

    ### COT4:
    # {
    #     'name': 'CoT4 new, egocentric only',
    #     'path': './logs/qwen_new_qn3_cot4_eval/Qwen3-VL-2B-Instruct/20260323_201740_responses_fixed.json',
    #     'tags': ['baseline'],
    # },
    # {
    #     'name': 'CoT4 new, overlay',
    #     'path': './logs/qwen_new_qn3_cot4_eval/Qwen3-VL-2B-Instruct/20260323_221648_responses_fixed.json',
    #     'tags': ['with_overlay'],
    # },
    # {
    #     'name': 'CoT4 new, demographics',
    #     'path': './logs/qwen_new_qn3_cot4_eval/Qwen3-VL-2B-Instruct/20260324_032710_responses_fixed.json',
    #     'tags': ['with_demographics'],
    # }
]



    # 'Qwen (Egocentric only, no CoT)': './logs/qwen_eval/Qwen3-VL-2B-Instruct/questionlast/20260310_115358_responses.json', # egocentric only, no CoT, qn0
    # 'Qwen (Egocentric only, no CoT)': './logs/qwen_qn3_direct_eval/Qwen3-VL-2B-Instruct/20260317_160241_responses.json', # egocentric only, no CoT, qn3
    # 'Qwen (Egocentric only, CoT)': './logs/qwen_cot_eval/Qwen3-VL-2B-Instruct/20260316_140731_responses_fixed.json', # egocentric only, CoT1, qn0
    # 'Qwen (Egocentric only, CoT)': './logs/qwen_qn3_cot1_eval/Qwen3-VL-2B-Instruct/20260317_161312_responses.json', # egocentric only, CoT1, qn3
    # 'Qwen (Egocentric + Gaze overlaid, CoT)': './logs/qwen_cot_eval/Qwen3-VL-2B-Instruct/20260317_110432_responses.json', # egocentric + gaze, CoT1, qn0
    # 'Qwen (Egocentric + Gaze overlaid, CoT)': './logs/qwen_cot3_eval/Qwen3-VL-2B-Instruct/20260317_140413_responses_fixed.json', # egocentric only, CoT3, qn0
    # 'Qwen (Egocentric + Gaze overlaid, CoT)': './logs/qwen_qn3_cot2_eval/Qwen3-VL-2B-Instruct/20260317_193128_responses.json', # egocentric only, CoT2, qn3


# Define which comparisons to generate (between pairs of models)
# Format: ('baseline_tag', 'other_tag') means compare models with baseline_tag vs other_tag
COMPARISON_RULES = [
    # ('baseline', 'with_cot1'),
    # ('baseline', 'with_cot4'),
    # ('with_cot1', 'with_cot4'),
    # ('baseline', 'with_cot6'),
    # ('baseline', 'with_cot5'),
    # ('with_cot6', 'with_cot5'),
    ('baseline', 'with_overlay'),
    ('baseline', 'with_demographics'),
    ('with_overlay', 'with_demographics'),
]

# ============================================================================
# LOAD MODELS DYNAMICALLY
# ============================================================================
# Create simple names (Model 1, Model 2, etc.) and mapping to full names
model_simple_names = {}  # Maps simple name to full name
model_full_names = {}    # Maps full name to simple name
json_files = {}

for idx, model_config in enumerate(MODELS_CONFIG, 1):
    simple_name = f"Model {idx}"
    full_name = model_config['name']
    model_simple_names[simple_name] = full_name
    model_full_names[full_name] = simple_name
    json_files[simple_name] = model_config['path']

# Load all JSON data using simple names
loaded_data = {}
for simple_name, json_path in json_files.items():
    try:
        with open(json_path) as f:
            loaded_data[simple_name] = json.load(f)
        print(f"Loaded {simple_name} ({model_simple_names[simple_name]}): {len(loaded_data[simple_name])} samples")
    except FileNotFoundError:
        print(f"Warning: {json_path} not found")
    except Exception as e:
        print(f"Error loading {json_path}: {e}")

# Create a mapping of video_id to data for each model
video_data_by_model = {}
for simple_name, data_list in loaded_data.items():
    video_data_by_model[simple_name] = {item['video_id']: item for item in data_list}

# Get all unique video IDs
all_video_ids = set()
for model_data in video_data_by_model.values():
    all_video_ids.update(model_data.keys())
all_video_ids = sorted(list(all_video_ids))

def find_video_file(video_id, video_dir):
    """Find video file in the given directory"""
    video_path = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        potential_path = Path(video_dir) / f"{video_id}{ext}"
        if potential_path.exists():
            video_path = str(potential_path)
            break
    return video_path

def highlight_differences(text1, text2):
    """Highlight differences between two texts in blue"""
    # Always return a tuple: (highlighted_text1, highlighted_text2)
    # If texts are identical: show full baseline and collapse current to '...'
    if text1 == text2:
        return text1, "<span style='font-weight:bold;'>...</span>"

    # If baseline appears anywhere inside the current text, collapse that matching
    # region in the current text to '...' to avoid repeating the same long text.
    try:
        if text1 and text1 in text2:
            # Replace the first occurrence only
            replaced = text2.replace(text1, "<span style='font-weight:bold;'>...</span>", 1)
            return text1, replaced

        # If current is contained in baseline, collapse the matching region in baseline
        if text2 and text2 in text1:
            replaced_baseline = text1.replace(text2, "<span style='font-weight:bold;'>...</span>", 1)
            return replaced_baseline, text2
    except Exception:
        # Fall back to naive behaviour if substring checks fail
        pass

    # Fallback: word-by-word alignment and highlighting (best-effort)
    words1 = text1.split()
    words2 = text2.split()

    result1 = []
    result2 = []

    max_len = max(len(words1), len(words2))
    for i in range(max_len):
        w1 = words1[i] if i < len(words1) else ""
        w2 = words2[i] if i < len(words2) else ""

        if w1 != w2:
            # Only highlight non-empty differing tokens to avoid excessive markup
            token1 = w1 if w1 else ""
            token2 = w2 if w2 else ""
            result1.append(f"<span style='background-color: #ADD8E6; font-weight: bold;'>{token1}</span>")
            result2.append(f"<span style='background-color: #ADD8E6; font-weight: bold;'>{token2}</span>")
        else:
            result1.append(w1)
            result2.append(w2)

    return " ".join(result1), " ".join(result2)

def display_video_comparison(video_id):
    """Display videos and model predictions for a given video_id"""
    
    # Find video files in both directories
    video_path_clips = find_video_file(video_id, './data/videodata_256/clips')
    video_path_overlay = find_video_file(video_id, './data/videodata_256/clips_dot')
    
    # Get ground truth answer and questions from all models
    ground_truth = None
    questions_dict = {}
    for model_name in sorted(loaded_data.keys()):
        if video_id in video_data_by_model[model_name]:
            item = video_data_by_model[model_name][video_id]
            if 'answer' in item and ground_truth is None:
                ground_truth = item['answer']
            if 'question' in item:
                questions_dict[model_name] = item['question']
    
    # Build right panel HTML with video ID, ground truth, and model comparisons
    comparison_html = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; height: 100%; overflow-y: auto;">
    <h2 style="margin-top: 0;">Video ID: {video_id}</h2>
    """
    
    # Display ground truth answer
    if ground_truth:
        comparison_html += f"""
        <div style='padding: 12px; background-color: #d4edda; border-radius: 5px; margin-bottom: 20px;'>
        <p style='margin: 0;'><b>Correct Answer:</b> <span style='color: green; font-weight: bold; font-size: 16px;'>{ground_truth}</span></p>
        </div>
        """
    
    comparison_html += "<hr style='margin: 20px 0;'>"
    
    # Order models based on MODELS_CONFIG order - using simple names (Model 1, Model 2, etc.)
    model_order = []  # Will hold simple names in MODELS_CONFIG order
    for idx, config in enumerate(MODELS_CONFIG, 1):
        simple_name = f"Model {idx}"
        if simple_name in loaded_data:
            model_order.append(simple_name)
    
    # Create comparison in columns for each model - dynamically set grid columns
    num_models = len(model_order)
    grid_cols = f"1fr " * num_models
    comparison_html += f"""
    <div style='display: grid; grid-template-columns: {grid_cols.strip()}; gap: 15px;'>
    """
    
    # Find the baseline model (first one in MODELS_CONFIG)
    baseline_model = model_order[0] if model_order else None
    baseline_question = questions_dict.get(baseline_model)
    
    for simple_name in model_order:
        if video_id not in video_data_by_model[simple_name]:
            comparison_html += f"""
            <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>
            <h4 style='margin-top: 0;'>{simple_name}: {model_simple_names[simple_name]}</h4>
            <p>No data available</p>
            </div>
            """
            continue
        
        item = video_data_by_model[simple_name][video_id]
        
        comparison_html += f"""
        <div style='padding: 12px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;'>
        <h4 style='margin: 0 0 10px 0; color: #2c3e50;'>{simple_name}: {model_simple_names[simple_name]}</h4>
        """
        
        # Question with difference highlighting (compared against baseline)
        if 'question' in item:
            question_text = item['question']
            
            # Highlight differences only if this is NOT the baseline model
            if baseline_model and baseline_model != simple_name:
                # Compare against baseline
                if baseline_question and baseline_question != question_text:
                    # Questions are different - highlight them
                    try:
                        highlighted_baseline, highlighted_current = highlight_differences(
                            baseline_question, 
                            question_text
                        )
                        comparison_html += f"<p style='margin: 8px 0;'><b>Q:</b> {highlighted_current}</p>"
                    except Exception as e:
                        # If highlighting fails, just show the text
                        comparison_html += f"<p style='margin: 8px 0;'><b>Q:</b> {question_text}</p>"
                elif baseline_question and baseline_question == question_text:
                    comparison_html += f"<p style='margin: 8px 0;'><b>Q:</b> <span style='font-weight:bold;'>...</span></p>"
                else:
                    # Questions are the same or baseline_question is None
                    comparison_html += f"<p style='margin: 8px 0;'><b>Q:</b> {question_text}</p>"
            else:
                # This is the baseline model - replace the full question with an ellipsis
                comparison_html += f"<p style='margin: 8px 0;'><b>Q:</b> {baseline_question} </p>"
        
        # Model prediction
        if 'pred_answer' in item:
            pred = item['pred_answer']
            is_correct = pred == item.get('answer', '')
            color = 'green' if is_correct else 'red'
            status = '✓' if is_correct else '✗'
            comparison_html += f"<p style='margin: 8px 0;'><b>A:</b> <span style='color: {color}; font-weight: bold;'>{pred}</span> <span style='color: {color};'>{status}</span></p>"
        
        # Reasoning (if available)
        if 'reasoning' in item:
            comparison_html += f"<p style='margin: 8px 0; color: #555;'><b>Reasoning:</b> {item['reasoning']}</p>"
        
        comparison_html += "</div>"
    
    comparison_html += """
    </div>
    </div>
    """
    
    return video_path_clips, video_path_overlay, comparison_html



def df_compare():
    """
    Build a comparison DataFrame for all models dynamically based on MODELS_CONFIG.
    Generates pairwise comparisons based on COMPARISON_RULES.
    """
    
    model_names = list(loaded_data.keys())
    
    # Helper function to get model config by name
    def get_model_config(name: str) -> Dict[str, Any]:
        for config in MODELS_CONFIG:
            if config['name'] == name:
                return config
        return None
    
    # Build DataFrames for each model
    def df_from_model(model_name):
        rows = []
        for item in loaded_data.get(model_name, []):
            rows.append({
                'video_id': item.get('video_id'),
                'answer': item.get('answer', ''),
                'pred_answer': item.get('pred_answer', ''),
            })
        return pd.DataFrame(rows)
    
    # Start with the first model
    if not model_names:
        return None
    
    df = df_from_model(model_names[0]).rename(columns={
        'pred_answer': f'pred_{model_names[0]}',
        'answer': f'answer_{model_names[0]}'
    })
    
    # Merge all other models
    for model_name in model_names[1:]:
        df_model = df_from_model(model_name).rename(columns={
            'pred_answer': f'pred_{model_name}',
            'answer': f'answer_{model_name}'
        })
        df = pd.merge(df, df_model, on='video_id', how='outer')
    
    # Use the first model's answer as ground truth if available
    answer_cols = [col for col in df.columns if col.startswith('answer_')]
    if answer_cols:
        df['answer'] = df[answer_cols[0]]
    else:
        df['answer'] = ''
    
    # Fill NaN values with empty string for comparisons
    pred_cols = [col for col in df.columns if col.startswith('pred_')]
    for col in pred_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Generate pairwise comparisons based on COMPARISON_RULES
    comparison_columns = {}
    
    for baseline_tag, other_tag in COMPARISON_RULES:
        # Find models with these tags (use indices to map to simple names)
        baseline_models = []
        other_models = []
        
        for idx, config in enumerate(MODELS_CONFIG, 1):
            simple_name = f"Model {idx}"
            if simple_name in loaded_data:  # Only if model was loaded
                if baseline_tag in config.get('tags', []):
                    baseline_models.append(simple_name)
                if other_tag in config.get('tags', []):
                    other_models.append(simple_name)
        
        # Create comparisons for each pair
        for baseline_simple_name in baseline_models:
            for other_simple_name in other_models:
                if baseline_simple_name == other_simple_name:
                    continue
                
                baseline_pred_col = f'pred_{baseline_simple_name}'
                other_pred_col = f'pred_{other_simple_name}'
                
                # Only create comparison if both columns exist
                if baseline_pred_col in df.columns and other_pred_col in df.columns:
                    # baseline model is better: baseline correct, other wrong
                    comp_col_baseline_better = f'{baseline_simple_name}_vs_{other_simple_name}_baseline_better'
                    df[comp_col_baseline_better] = (
                        (df[baseline_pred_col] == df['answer']) & 
                        (df[other_pred_col] != df['answer'])
                    )
                    comparison_columns[comp_col_baseline_better] = {
                        'baseline': baseline_simple_name,
                        'other': other_simple_name,
                        'winner': baseline_simple_name
                    }
                    
                    # other model is better: other correct, baseline wrong
                    comp_col_other_better = f'{other_simple_name}_vs_{baseline_simple_name}_other_better'
                    df[comp_col_other_better] = (
                        (df[other_pred_col] == df['answer']) & 
                        (df[baseline_pred_col] != df['answer'])
                    )
                    comparison_columns[comp_col_other_better] = {
                        'baseline': baseline_simple_name,
                        'other': other_simple_name,
                        'winner': other_simple_name
                    }
    
    return df, comparison_columns

# Compute the comparison DataFrame and seven selection lists
try:
    result = df_compare()
    if result is not None:
        df_comp, comparison_columns = result
    else:
        df_comp = None
        comparison_columns = {}
except Exception as e:
    print(f"Error in df_compare: {e}")
    df_comp = None
    comparison_columns = {}

# Extract video ID lists for each comparison
comparison_lists = {}
if df_comp is not None and not df_comp.empty:
    all_video_ids = sorted(df_comp['video_id'].dropna().astype(str).tolist())
    
    for comp_col, comp_info in comparison_columns.items():
        if comp_col in df_comp.columns:
            video_ids = sorted(
                df_comp.loc[df_comp[comp_col] == True, 'video_id'].astype(str).tolist()
            )
            comparison_lists[comp_col] = video_ids
else:
    # safe fallback
    all_video_ids = sorted(list(all_video_ids))
    comparison_lists = {}


# Create Gradio interface
with gr.Blocks(
    title="Video Model Comparison",
    theme=gr.themes.Soft(),
    head="""
    <style>
        :root {
            --light-bg: #ffffff;
            --light-text: #000000;
            --light-border: #ddd;
            --light-panel: #f9f9f9;
            --dark-bg: #1e1e1e;
            --dark-text: #ffffff;
            --dark-border: #444;
            --dark-panel: #2d2d2d;
        }
    </style>
    """
) as demo:
    # Theme toggle at the top
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown("# Video Model Comparison")
        with gr.Column(scale=1):
            theme_toggle = gr.Button("🌙 Night Mode", variant="secondary")
    
    gr.Markdown("Select a video ID to compare model predictions")
    
    # Create tabs for different video selections dynamically
    tabs_and_dropdowns = []
    
    with gr.Tabs():
        # Tab 1: All Videos
        with gr.TabItem("1. All Videos"):
            all_videos_dropdown = gr.Dropdown(
                choices=all_video_ids,
                label="All Video IDs",
                value=all_video_ids[0] if all_video_ids else None,
                interactive=True
            )
            tabs_and_dropdowns.append(all_videos_dropdown)
        
        # Tabs 2+: Dynamic comparison tabs
        tab_index = 2
        for comp_col, comp_info in comparison_columns.items():
            if comp_col in comparison_lists:
                video_ids = comparison_lists[comp_col]
                
                # Create tab label: "Model X > Model Y" or "Model Y > Model X"
                baseline_simple = comp_info['baseline']
                other_simple = comp_info['other']
                winner = comp_info['winner']
                
                if winner == baseline_simple:
                    tab_label = f"{tab_index}. {baseline_simple} > {other_simple}"
                    dropdown_label = f"{baseline_simple} works better than {other_simple}"
                else:
                    tab_label = f"{tab_index}. {other_simple} > {baseline_simple}"
                    dropdown_label = f"{other_simple} works better than {baseline_simple}"
                
                with gr.TabItem(tab_label):
                    dropdown = gr.Dropdown(
                        choices=video_ids,
                        label=dropdown_label,
                        value=video_ids[0] if video_ids else None,
                        interactive=True
                    )
                    tabs_and_dropdowns.append(dropdown)
                
                tab_index += 1
    
    # Create a state variable to track the selected video ID
    selected_video_id = gr.State(value=all_video_ids[0] if all_video_ids else None)
    
    # Main layout: Left (videos stacked) and Right (predictions)
    with gr.Row():
        # Left column: Videos stacked vertically
        with gr.Column(scale=1):
            gr.Markdown("### Videos")
            video_player_clips = gr.Video(
                label="Original",
                interactive=False,
                elem_classes="video-container"
            )
            video_player_overlay = gr.Video(
                label="Gaze Overlaid",
                interactive=False,
                elem_classes="video-container"
            )
        
        # Right column: Predictions
        with gr.Column(scale=3):
            comparison_output = gr.HTML(
                value="Select a video to see predictions"
            )
    
    # Theme toggle button handler (JavaScript-based, but we can add a dummy function)
    theme_toggle.click(
        fn=lambda: None  # Theme is handled entirely by JavaScript
    )
    
    # Attach change events to all dropdown tabs
    for dropdown in tabs_and_dropdowns:
        dropdown.change(
            fn=display_video_comparison,
            inputs=dropdown,
            outputs=[video_player_clips, video_player_overlay, comparison_output]
        )
    
    # Initial load - load the first video from the first tab
    if all_video_ids and tabs_and_dropdowns:
        tabs_and_dropdowns[0].value = all_video_ids[0]


if __name__ == "__main__":
    # Add custom JavaScript to enable autoplay and theme switching
    demo.launch(
        share=False, 
        show_error=True,
        css="""
        .video-container { display: flex; gap: 10px; width: 100%; }
        video { width: 100%; height: auto; max-width: 400px; }
        
        /* Light Mode (Default) */
        body, .gradio-container {
            background-color: #ffffff;
            color: #000000;
        }
        
        .gradio-box, .gradio-panel {
            background-color: #f9f9f9;
            border-color: #ddd;
        }
        
        /* Dark Mode */
        body.dark-mode, .dark-mode .gradio-container {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        .dark-mode .gradio-box,
        .dark-mode .gradio-panel,
        .dark-mode [class*="panel"],
        .dark-mode .gradio-tabitem {
            background-color: #2d2d2d;
            border-color: #444;
        }
        
        .dark-mode input,
        .dark-mode textarea,
        .dark-mode select {
            background-color: #3a3a3a;
            color: #ffffff;
            border-color: #555;
        }
        
        .dark-mode button {
            background-color: #404040;
            color: #ffffff;
            border-color: #555;
        }
        
        .dark-mode .gradio-label {
            color: #ffffff;
        }
        
        .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4 {
            color: #ffffff;
        }
        
        .dark-mode .gradio-dropdown > div {
            background-color: #3a3a3a;
            border-color: #555;
        }
        
        .dark-mode .gradio-dropdown-arrow {
            color: #ffffff;
        }
        """,
        head="""
        <script>
            let themeInitialized = false;
            let autoplayInitialized = false;
            
            function enableAutoplay() {
                if (autoplayInitialized) return;
                document.querySelectorAll('video').forEach(video => {
                    video.autoplay = true;
                    video.muted = true;
                    video.play().catch(err => console.log('Autoplay prevented:', err));
                });
                autoplayInitialized = true;
            }
            
            function toggleTheme() {
                const isDarkMode = document.body.classList.toggle('dark-mode');
                localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');
                updateThemeButton();
            }
            
            function updateThemeButton() {
                const buttons = document.querySelectorAll('button');
                buttons.forEach(btn => {
                    if (btn.textContent.includes('🌙') || btn.textContent.includes('☀️')) {
                        if (document.body.classList.contains('dark-mode')) {
                            btn.textContent = btn.textContent.replace('🌙 Night Mode', '☀️ Day Mode').replace('Night', 'Day');
                        } else {
                            btn.textContent = btn.textContent.replace('☀️ Day Mode', '🌙 Night Mode').replace('Day', 'Night');
                        }
                    }
                });
            }
            
            function initTheme() {
                if (themeInitialized) return;
                const savedTheme = localStorage.getItem('theme') || 'light';
                if (savedTheme === 'dark') {
                    document.body.classList.add('dark-mode');
                }
                updateThemeButton();
                
                setTimeout(() => {
                    const buttons = document.querySelectorAll('button');
                    buttons.forEach(btn => {
                        if (btn.textContent.includes('🌙') || btn.textContent.includes('☀️')) {
                            if (!btn.classList.contains('theme-toggle-initialized')) {
                                btn.addEventListener('click', toggleTheme);
                                btn.classList.add('theme-toggle-initialized');
                            }
                        }
                    });
                }, 100);
                themeInitialized = true;
            }
            
            window.addEventListener('load', function() {
                initTheme();
                enableAutoplay();
            });
            
            document.addEventListener('DOMContentLoaded', () => {
                initTheme();
                enableAutoplay();
            });
        </script>
        """
    )
