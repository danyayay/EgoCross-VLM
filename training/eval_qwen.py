import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info


from utils.analyze_groundvqa_results import analyze, pretty_print_report
from utils.qwen_utils import build_message
from utils.util import enable_strict_determinism, setup_logging
from utils.parser.extract_answers import extract_answer_from_output


# ── Shared eval utilities (model-agnostic) ───────────────────────────────
# Imported here so that callers using `from training.eval_qwen import ...`
# continue to work without modification.
from utils.eval_utils import (
    _COT_PROMPTS,
    _QUESTION_STARTERS,
    _split_context_question,
    build_prompt_from_annotation,
    infer_task_from_path,
    get_video_frames,
    _parse_model_output,
)

def inference(model, processor, video, prompt,
              total_pixels=20480 * 32 * 32, min_pixels=64 * 32 * 32,
              max_frames=2048, sample_fps=2, temperature=None, max_new_tokens=None,
              no_video=False):
    """Run multimodal inference and return (output_text, inference_meta)."""
    if no_video:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = build_message(video, prompt, total_pixels, min_pixels, max_frames, sample_fps)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       video_metadata=video_metadatas, **video_kwargs,
                       do_resize=False, return_tensors="pt")
    inputs = inputs.to('cuda')

    start_time = time.perf_counter()
    output_ids = model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens)
    duration = time.perf_counter() - start_time

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    gen_token_count = sum(len(ids) for ids in generated_ids)
    tokens_per_second = gen_token_count / duration if duration > 0 else float('inf')
    is_finished = processor.tokenizer.eos_token_id == generated_ids[-1][-1]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)[0]
    inference_meta = {
        'is_finished': is_finished.item(),
        'duration': duration,
        'token_count': gen_token_count,
        'tokens_per_sec': tokens_per_second,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
    }
    return output_text, inference_meta



def evaluate_on_annotations(model, processor, logname, ann_file, args):
    annos = json.loads(Path(ann_file).read_text())
    task_name = infer_task_from_path(ann_file)
    results = []
    start_time = time.time()
    
    # model for answer extraction
    extractor_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    extractor_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    inference_time = 0

    for anno in tqdm(annos, desc='Evaluating'):
        video_id = anno.get('video_id')
        video_path = os.path.join(args.video_root, video_id + '.mp4')
        if not os.path.exists(video_path):
            results.append({'video_id': video_id, 'error': 'video_not_found'})
            continue

        prompt_text, _ctx, _qst, options_map, correct_answer = build_prompt_from_annotation(
            anno, cot_type=args.cot_type, prompt_variant=getattr(args, 'prompt_variant', 'p6'),
            task_name=task_name, context_features=getattr(args, 'context_features', 'none'),
            context_feature_fps=getattr(args, 'context_feature_fps', 'auto'),
            num_frames=getattr(args, 'num_frames', None))
        try:
            output_text, inference_meta = inference(
                model, processor, video_path, prompt_text,
                max_frames=args.max_frames, total_pixels=args.total_pixels,
                min_pixels=args.min_pixels, sample_fps=args.sample_fps,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                no_video=args.no_video)
            retry_count = 0
            while not inference_meta['is_finished'] and retry_count <= 2:
                retry_count += 1
                logging.info('Output truncated for video_id=%s, retrying (attempt %d)', video_id, retry_count)
                output_text, inference_meta = inference(
                    model, processor, video_path, prompt_text,
                    max_frames=args.max_frames, total_pixels=args.total_pixels,
                    min_pixels=args.min_pixels, sample_fps=args.sample_fps,
                    temperature=0.1 * retry_count,
                    max_new_tokens=args.max_new_tokens * retry_count,
                    no_video=args.no_video)
        except Exception:
            logging.exception("Runtime error while processing video_id=%s", video_id)
            results.append({'video_id': video_id, 'error': 'runtime_error'})
            continue

        pred_answer, reasoning = _parse_model_output(output_text, options_map, args.cot_type)
        if inference_meta['is_finished']:
            valid_labels = [v.lower() for v in options_map.values()]
            if pred_answer not in valid_labels:
                pred_answer = extract_answer_from_output(
                    extractor_model, extractor_tokenizer, output_text, valid_labels=valid_labels)
                reasoning = output_text
        else:
            pred_answer = 'unknown'

        results.append({
            'video_id': video_id,
            'question': prompt_text,
            'task_name': task_name,
            'context_features': getattr(args, 'context_features', 'none'),
            'context_feature_fps': getattr(args, 'context_feature_fps', 'auto'),
            'options': options_map,
            'answer': correct_answer,
            'pred_answer': pred_answer,
            'reasoning': reasoning,
            'output_text': output_text,
            'is_finished': inference_meta['is_finished'],
            'duration': inference_meta['duration'],
            'token_count': inference_meta['token_count'],
            'tokens_per_sec': inference_meta['tokens_per_sec'],
            'temperature': inference_meta['temperature'],
            'max_new_tokens': inference_meta['max_new_tokens'],
        })
        inference_time += inference_meta['duration']

    logging.info("Inference time: %.2f seconds", inference_time)
    logging.info("Average inference time per sample: %.2f seconds", inference_time / len(annos) if annos else 0)

    os.makedirs(args.outdir, exist_ok=True)
    output_filename = f'{logname}_responses.json'
    output_path = os.path.join(args.outdir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d results to %s", len(results), output_path)

    report = analyze(output_path)
    pretty_print_report(report)
    report_path = os.path.join(args.outdir, output_filename.split('.')[0] + '_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info(
        "Samples: %d, Correct: %d/%d, Accuracy: %.4f, Macro F1: %.4f",
        report['num_samples'],
        report['classification']['correct'],
        report['classification']['total'],
        report['classification']['accuracy'],
        report['classification']['macro_f1'],
    )
    logging.info("Wrote report to %s", report_path)


def main():
    parser = argparse.ArgumentParser(description='Batch evaluate Qwen video model on annotated videos')
    parser.add_argument('--ann_file',
                        default='features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json')
    parser.add_argument('--video_root', default='data/videodata_256/clips')
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--log_dir', default='logs/qwen_eval_new_qn3')
    parser.add_argument('--cache_dir', default='.cache', help='Directory for cached video downloads and frame files')

    parser.add_argument('--model_name', default='Qwen/Qwen3-VL-2B-Instruct',
                        choices=['Qwen/Qwen3-VL-2B-Instruct', 'Qwen/Qwen3-VL-4B-Instruct',
                                 'Qwen/Qwen3-VL-4B-Instruct-FP8'])
    parser.add_argument('--lora_adapter', default=None,
                        help='Path to PEFT LoRA adapter (optional)')
    parser.add_argument('--cot_type', default='cot4',
                        choices=['none', 'cot1', 'cot2', 'cot3', 'cot4', 'cot5', 'cot6', 'cot7'],
                        help='Chain-of-thought prompt variant (none = plain multiple-choice)')
    parser.add_argument('--no_video', action='store_true',
                        help='Language-only baseline: send text prompt only, no video frames')
    parser.add_argument('--prompt_variant', default='p6',
                        choices=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'])
    parser.add_argument('--context_features', default='none')
    parser.add_argument('--context_feature_fps', default='auto')

    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--total_pixels', type=int, default=20480 * 32 * 32)
    parser.add_argument('--min_pixels', type=int, default=64 * 32 * 32)
    parser.add_argument('--max_frames', type=int, default=2048)
    parser.add_argument('--sample_fps', type=int, default=2)

    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if args.cot_type == 'none':
        args.cot_type = None

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_dir = os.path.join(args.log_dir+'_'+str(args.cot_type), args.model_name.split('/')[-1])
    if args.outdir:
        args.outdir = os.path.join(args.outdir, args.model_name.split('/')[-1])
    else:
        args.outdir = args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{ts}.log')
    setup_logging(log_file)
    logging.info("Evaluation arguments: %s", vars(args))

    processor = AutoProcessor.from_pretrained(args.model_name)
    model, output_loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto", output_loading_info=True)

    generation_style = {"max_new_tokens": args.max_new_tokens,
                        "repetition_penalty": args.repetition_penalty}
    if args.temperature == 0.0:
        generation_style.update({'do_sample': False, 'temperature': 0.0, 'top_p': None, 'top_k': None})
    else:
        if args.temperature is not None:
            generation_style["temperature"] = args.temperature
        if args.top_p is not None:
            generation_style["top_p"] = args.top_p
        if args.repetition_penalty is not None:
            generation_style["repetition_penalty"] = args.repetition_penalty
    model.generation_config.update(**generation_style)
    logging.info("model.generation_config: %s", model.generation_config)
    logging.info("output_loading_info: %s", output_loading_info)

    enable_strict_determinism(args.seed)

    if args.lora_adapter:
        logging.info("Loading LoRA adapter from: %s", args.lora_adapter)
        model = PeftModel.from_pretrained(model, args.lora_adapter, device_map='auto')
        try:
            model = model.to('cuda')
        except Exception:
            pass

    log_name = args.lora_adapter.split('/')[-1] if args.lora_adapter else ts
    evaluate_on_annotations(model, processor, log_name, args.ann_file, args=args)


if __name__ == '__main__':
    main()
