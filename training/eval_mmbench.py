"""Evaluate a (optionally LoRA-adapted) Qwen3-VL model on MMBench.

Purpose: sanity-check whether fine-tuning on the pedestrian-intention task
degrades general multimodal ability, by measuring multiple-choice accuracy on
a stratified sample of MMBench (English dev split) questions.

Data source: HuggingFace ``lmms-lab/MMBench`` (images decoded to PIL already,
no manual base64/TSV parsing needed).

Typical usage — base model:
    python -m training.eval_mmbench --outdir results/mmbench

Same sampled questions, LoRA-adapted model (pass the same --seed/--samples_per_category
so both runs draw an identical subset):
    python -m training.eval_mmbench --outdir results/mmbench \
        --lora_adapter logs/qwen_training_intention/Qwen3-VL-2B-Instruct/<timestamp>/best-...
"""

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from utils.eval_utils import _match_option_letter, _match_option_text
from utils.util import enable_strict_determinism, setup_logging

_OPTION_KEYS = ("A", "B", "C", "D")

# The HF dataset column is "L2-category" (capital L2); we expose the friendlier
# "l2-category" spelling everywhere else (CLI flag, report keys).
_DATASET_COLUMN_FOR_LEVEL = {"category": "category", "l2-category": "L2-category"}


def load_mmbench(split: str = "dev"):
    """Load the English MMBench split from HuggingFace as a list of dicts."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/MMBench", "en", split=split)
    return list(ds)


def stratified_sample(rows: list, samples_per_category: int, category_level: str, seed: int) -> list:
    """Sample up to ``samples_per_category`` rows per category/l2-category group.

    Groups with fewer available rows contribute all of them (no upsampling).
    """
    assert category_level in _DATASET_COLUMN_FOR_LEVEL, category_level
    column = _DATASET_COLUMN_FOR_LEVEL[category_level]
    rng = random.Random(seed)
    by_group: dict[str, list] = {}
    for row in rows:
        by_group.setdefault(row[column], []).append(row)

    sampled = []
    for group in sorted(by_group):
        pool = by_group[group]
        rng.shuffle(pool)
        sampled.extend(pool[:samples_per_category])
    rng.shuffle(sampled)
    return sampled


def build_mmbench_prompt(row: dict) -> tuple[str, dict]:
    """Build an MCQ prompt string and an options_map ({'A': text, ...}) for one row."""
    options_map = {k: row[k] for k in _OPTION_KEYS if row.get(k) not in (None, "", "nan")}
    options_str = " ".join(f"({k}) {v}" for k, v in options_map.items())
    parts = []
    if row.get("hint") and str(row["hint"]).strip().lower() not in ("", "nan"):
        parts.append(str(row["hint"]).strip())
    parts.append(str(row["question"]).strip())
    parts.append(f"Choose one option: {options_str}.")
    parts.append("Answer with the correct option only.")
    prompt_text = " ".join(parts)
    return prompt_text, options_map


def inference_on_image(model, processor, image, prompt_text: str,
                        max_new_tokens: int = 64, temperature: float = 0.0):
    """Run single-image multimodal inference and return the decoded output text."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = inputs.to("cuda")

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if temperature == 0.0:
        gen_kwargs.update({"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
    else:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output_text


_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")


def extract_timestamp_from_path(path: str) -> str | None:
    """Return the first YYYYMMDD_HHMMSS-style timestamp found in a checkpoint path, if any."""
    for part in Path(path).parts:
        match = _TIMESTAMP_RE.search(part)
        if match:
            return match.group(0)
    return None


def answer_letter_for_option(options_map: dict, answer_value) -> str | None:
    """MMBench 'answer' is already a letter (A/B/C/D); pass through if valid."""
    answer_value = str(answer_value).strip().upper()
    return answer_value if answer_value in options_map else None


def evaluate(model, processor, rows: list, args, logname: str) -> dict:
    results = []
    inference_time = 0.0

    for row in rows:
        options_map = {k: row[k] for k in _OPTION_KEYS if row.get(k) not in (None, "", "nan")}
        prompt_text, _ = build_mmbench_prompt(row)
        gt_letter = answer_letter_for_option(options_map, row["answer"])

        start = time.perf_counter()
        try:
            output_text = inference_on_image(
                model, processor, row["image"], prompt_text,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        except Exception:
            logging.exception("Runtime error on index=%s", row.get("index"))
            results.append({"index": row.get("index"), "error": "runtime_error"})
            continue
        duration = time.perf_counter() - start
        inference_time += duration

        letter_options = {k: k for k in options_map}
        pred_letter = _match_option_letter(output_text, letter_options)
        if pred_letter is not None:
            pred_letter = pred_letter.upper()
        if pred_letter is None:
            pred_text = _match_option_text(output_text, options_map)
            if pred_text is not None:
                pred_letter = next(
                    (k for k, v in options_map.items() if str(v).lower() == pred_text), None)

        results.append({
            "index": row.get("index"),
            "category": row.get("category"),
            "l2-category": row.get("L2-category"),
            "question": row.get("question"),
            "options": options_map,
            "gt_answer": gt_letter,
            "pred_answer": pred_letter,
            "output_text": output_text,
            "correct": pred_letter is not None and pred_letter == gt_letter,
            "duration": duration,
        })

    logging.info("Inference time: %.2f seconds over %d samples", inference_time, len(rows))

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, f"{logname}_responses.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("Saved %d results to %s", len(results), output_path)

    report = build_report(results, category_level=args.category_level)
    report_path = os.path.join(args.outdir, f"{logname}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Overall accuracy: %.4f (%d/%d)",
                 report["overall"]["accuracy"], report["overall"]["correct"], report["overall"]["total"])
    for group, stats in sorted(report["per_category"].items()):
        logging.info("  %-30s acc=%.4f (%d/%d)", group, stats["accuracy"], stats["correct"], stats["total"])
    logging.info("Wrote report to %s", report_path)
    return report


def build_report(results: list, category_level: str = "category") -> dict:
    valid = [r for r in results if "error" not in r]
    total = len(valid)
    correct = sum(1 for r in valid if r["correct"])

    per_category: dict[str, dict] = {}
    for r in valid:
        group = r.get(category_level) or "unknown"
        stats = per_category.setdefault(group, {"correct": 0, "total": 0})
        stats["total"] += 1
        stats["correct"] += int(r["correct"])
    for stats in per_category.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    return {
        "num_samples": len(results),
        "num_errors": len(results) - total,
        "category_level": category_level,
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        },
        "per_category": per_category,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL (base or LoRA-adapted) on MMBench")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--category_level", default="l2-category", choices=["category", "l2-category"],
                        help="Granularity to stratify sampling and report accuracy by: "
                             "'category' (~20 fine-grained leaf skills) or 'l2-category' (~6-9 broad groups)")
    parser.add_argument("--samples_per_category", type=int, default=100,
                        help="Max samples drawn per category/l2-category group")
    parser.add_argument("--sampled_indices_file", default=None,
                        help="Optional path to a JSON list of MMBench 'index' values. "
                             "If given and exists, reuse this exact subset (for base-vs-finetuned parity) "
                             "instead of re-sampling. If given and missing, sample and write the indices there.")

    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    # parser.add_argument("--lora_adapter", default='logs/vlm_training/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260528_034756', help="Path to PEFT LoRA adapter (optional)")
    # parser.add_argument("--lora_adapter", default='logs/vlm_training_dot/seed_42/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260529_000601_ctxego_motion+gaze_direction_cfps4_preface_legacy', help="Path to PEFT LoRA adapter (optional)")
    parser.add_argument("--lora_adapter", default=None, help="Path to PEFT LoRA adapter (optional)")

    parser.add_argument("--outdir", default=None,
                        help="Where to write *_responses.json / *_report.json. "
                             "Defaults to --log_dir so logs and outputs live together.")
    parser.add_argument("--log_dir", default="logs/mmbench_eval")

    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--subset_seed", type=int, default=42)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.lora_adapter:
        ts = extract_timestamp_from_path(args.lora_adapter) or ts
        match = re.search(r"seed_(\d+)", args.lora_adapter)
        model_seed = f'model-seed-{match.group(1)}' if match else 'model-seed-unknown'
    else:
        model_seed = ''
    args.log_dir = os.path.join(args.log_dir, args.model_name.split("/")[-1], f"{args.category_level}_{args.samples_per_category}_subset-seed-{args.subset_seed}", model_seed)
    if args.outdir:
        args.outdir = os.path.join(args.outdir, args.model_name.split("/")[-1], f"{args.category_level}_{args.samples_per_category}_subset-seed-{args.subset_seed}", model_seed)
    else:
        args.outdir = args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(os.path.join(args.log_dir, f"{ts}.log"))
    logging.info("MMBench eval arguments: %s", vars(args))

    enable_strict_determinism(args.subset_seed)

    logging.info("Loading MMBench (%s) from HuggingFace...", args.split)
    rows = load_mmbench(split=args.split)
    logging.info("Loaded %d MMBench rows", len(rows))

    if args.sampled_indices_file and os.path.exists(args.sampled_indices_file):
        wanted = set(json.loads(Path(args.sampled_indices_file).read_text()))
        sampled = [r for r in rows if r.get("index") in wanted]
        logging.info("Reusing %d previously sampled indices from %s",
                     len(sampled), args.sampled_indices_file)
    else:
        sampled = stratified_sample(rows, args.samples_per_category, args.category_level, args.subset_seed)
        logging.info("Sampled %d rows (%s, %d per group)",
                     len(sampled), args.category_level, args.samples_per_category)
        if args.sampled_indices_file:
            os.makedirs(os.path.dirname(args.sampled_indices_file) or ".", exist_ok=True)
            Path(args.sampled_indices_file).write_text(
                json.dumps([r["index"] for r in sampled]))
            logging.info("Wrote sampled indices to %s", args.sampled_indices_file)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model, output_loading_info = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto", output_loading_info=True)
    logging.info("output_loading_info: %s", output_loading_info)

    if args.lora_adapter:
        logging.info("Loading LoRA adapter from: %s", args.lora_adapter)
        model = PeftModel.from_pretrained(model, os.path.abspath(args.lora_adapter))

    model.eval()
    log_name = Path(args.lora_adapter).name if args.lora_adapter else f"base_{ts}"
    evaluate(model, processor, sampled, args, logname=log_name)


if __name__ == "__main__":
    main()
