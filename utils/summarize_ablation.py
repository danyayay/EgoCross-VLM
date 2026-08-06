#!/usr/bin/env python3
import os
import re
import ast
import json
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_log_args_and_time(log_path, report_data):
    if not os.path.exists(log_path):
        return {}, None

    with open(log_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    args = {}
    for line in lines[:10]:
        if 'eval_vlm arguments:' in line:
            dict_part = line.split('eval_vlm arguments:')[1].strip()
            try:
                args = ast.literal_eval(dict_part)
            except Exception:
                # Regex fallback for key arguments
                m_model = re.search(r"'model_name':\s*'([^']+)'", dict_part)
                m_interleaved = re.search(r"'interleaved_timestamps':\s*(True|False)", dict_part)
                m_quantize = re.search(r"'quantize':\s*(True|False)", dict_part)
                m_prompt = re.search(r"'prompt_variant':\s*'([^']+)'", dict_part)
                m_frames = re.search(r"'num_frames':\s*(\d+)", dict_part)
                
                if m_model: args['model_name'] = m_model.group(1)
                if m_interleaved: args['interleaved_timestamps'] = m_interleaved.group(1) == 'True'
                if m_quantize: args['quantize'] = m_quantize.group(1) == 'True'
                if m_prompt: args['prompt_variant'] = m_prompt.group(1)
                if m_frames: args['num_frames'] = int(m_frames.group(1))
            break

    # Parse inference time
    inference_time = None
    for line in reversed(lines):
        if 'Total inference time:' in line:
            match = re.search(r'Total inference time:\s*([\d.]+)\s*s', line)
            if match:
                inference_time = float(match.group(1))
                break
        elif 'Inference time:' in line:
            match = re.search(r'Inference time:\s*([\d.]+)\s*seconds', line)
            if match:
                inference_time = float(match.group(1))
                break

    # GPT-4o or recovery report fallback
    if inference_time is None and report_data and 'recovery' in report_data:
        recovery = report_data['recovery']
        if 'run_window_start_unix' in recovery and 'run_window_end_unix' in recovery:
            inference_time = float(recovery['run_window_end_unix'] - recovery['run_window_start_unix'])

    # Start/end timestamp fallback
    if inference_time is None and len(lines) >= 2:
        try:
            def parse_ts(l):
                m = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', l)
                if m:
                    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                return None
            start_ts = None
            end_ts = None
            for l in lines:
                ts = parse_ts(l)
                if ts:
                    if start_ts is None:
                        start_ts = ts
                    end_ts = ts
            if start_ts and end_ts:
                inference_time = (end_ts - start_ts).total_seconds()
        except Exception:
            pass

    return args, inference_time

def main():
    search_dir = 'logs/vlm_eval_newformat'
    print(f"Scanning for report files in: {search_dir}")
    
    report_paths = glob.glob(os.path.join(search_dir, '**/*_report.json'), recursive=True)
    print(f"Found {len(report_paths)} report files.")

    rows = []
    for report_path in report_paths:
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except Exception as e:
            print(f"Error reading {report_path}: {e}")
            continue

        # Find matching log file
        # Usually report is name_responses_report.json or name_recovered_responses_report.json
        # Log is name.log in the same directory
        report_dir = os.path.dirname(report_path)
        report_name = os.path.basename(report_path)
        
        # Extract the timestamp prefix (e.g. 20260518_020605)
        prefix_match = re.match(r'^(\d{8}_\d{6})', report_name)
        if prefix_match:
            prefix = prefix_match.group(1)
            log_path = os.path.join(report_dir, f"{prefix}.log")
        else:
            # Fallback to any log file in the same directory
            log_files = glob.glob(os.path.join(report_dir, "*.log"))
            log_path = log_files[0] if log_files else ""

        # Parse log args and inference time
        args, inf_time = parse_log_args_and_time(log_path, report)

        # Retrieve arguments with folder name fallback
        model_full = args.get('model_name')
        if not model_full:
            # Fallback from path: logs/vlm_eval_newformat/Qwen3-VL-2B-Instruct/...
            parts = Path(report_path).parts
            if len(parts) >= 3:
                model_full = parts[2]
            else:
                model_full = "Unknown"

        # Make the model name short/clean
        model_name = model_full.replace('Qwen/', '').replace('OpenGVLab/', '')

        interleaved_val = args.get('interleaved_timestamps')
        if interleaved_val is None:
            interleaved = "Yes" if "interleaved" in report_dir else "No"
        else:
            interleaved = "Yes" if interleaved_val else "No"

        quantized_val = args.get('quantize')
        if quantized_val is None:
            quantized = "Yes" if "int8" in report_dir or "quantized" in report_dir else "No"
        else:
            quantized = "Yes" if quantized_val else "No"

        prompt = args.get('prompt_variant')
        if not prompt:
            prompt_match = re.search(r'_p(\d+)', report_dir)
            prompt = f"p{prompt_match.group(1)}" if prompt_match else "p0"

        frame_num = args.get('num_frames')
        if not frame_num:
            frame_match = re.search(r'_f(\d+)', report_dir)
            frame_num = int(frame_match.group(1)) if frame_match else 8

        # Extract CoT style from directory name (e.g., cotcot1_f8... -> cot1, cotnocot_f8... -> none)
        cot_style = "none"
        cot_match = re.search(r'cot(none|cot\d+)', report_dir)
        if cot_match:
            cot_style = cot_match.group(1)

        # Extract classification metrics
        classification = report.get('classification', {})
        accuracy = classification.get('accuracy')
        macro_f1 = classification.get('macro_f1')

        per_class = classification.get('per_class', {})
        cross_metrics = per_class.get('cross', {})
        yield_metrics = per_class.get('yield', {})

        cross_f1 = cross_metrics.get('f1')
        cross_recall = cross_metrics.get('recall')
        yield_f1 = yield_metrics.get('f1')
        yield_recall = yield_metrics.get('recall')

        # Add to row list
        rows.append({
            'Model': model_name,
            'CoT style': cot_style,
            'interleaved?': interleaved,
            'quantized?': quantized,
            'prompt': prompt,
            'frame_num': frame_num,
            'Accuracy': accuracy,
            'Macro F1': macro_f1,
            'Cross F1': cross_f1,
            'Yield F1': yield_f1,
            'Cross Recall': cross_recall,
            'Yield Recall': yield_recall,
            'Inference time': inf_time
        })

    if not rows:
        print("No ablation rows collected!")
        return

    df = pd.DataFrame(rows)

    # Sort logic: Model, CoT style, interleaved?, quantized?, prompt, frame_num
    # For custom sorting of prompts, we strip 'p' and convert to int
    def prompt_sort_key(p):
        try:
            return int(p.replace('p', ''))
        except Exception:
            return 99

    def cot_sort_key(cot):
        cot_order = {'none': 0, 'cot1': 1, 'cot4': 2, 'cot6': 3, 'cot7': 4}
        return cot_order.get(cot, 99)

    df['prompt_int'] = df['prompt'].apply(prompt_sort_key)
    df['cot_sort'] = df['CoT style'].apply(cot_sort_key)
    
    # Map interleaved/quantized to sorting friendly values
    df['interleaved_sort'] = df['interleaved?'].map({'Yes': 0, 'No': 1})
    df['quantized_sort'] = df['quantized?'].map({'Yes': 0, 'No': 1})

    df = df.sort_values(by=[
        'Model',
        'cot_sort',
        'interleaved_sort', 
        'quantized_sort', 
        'prompt_int', 
        'frame_num'
    ]).reset_index(drop=True)

    # Drop sorting columns
    df = df.drop(columns=['prompt_int', 'cot_sort', 'interleaved_sort', 'quantized_sort'])

    # Round metric columns for printing
    metric_cols = ['Accuracy', 'Macro F1', 'Cross F1', 'Yield F1', 'Cross Recall', 'Yield Recall']
    for col in metric_cols:
        df[col] = df[col].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
    
    if 'Inference time' in df.columns:
        df['Inference time'] = df['Inference time'].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    # Output to stdout as a beautiful markdown table
    print("\n=== Ablation Study Results Summary ===")
    print(df.to_markdown(index=False))

    # Save to Excel
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    excel_path = os.path.join(out_dir, 'ablation_study_summary.xlsx')
    
    # Save using pandas and openpyxl
    df.to_excel(excel_path, index=False, sheet_name='Ablation Results')
    print(f"\nSuccessfully summarized {len(df)} rows into Excel file: {excel_path}")

if __name__ == '__main__':
    main()
