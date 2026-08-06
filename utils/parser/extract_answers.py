import json
import os
import re
import sys
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from utils.analyze_groundvqa_results import analyze, pretty_print_report

model_name = "Qwen/Qwen3-VL-2B-Instruct"


def extract_answer_binary(output: str, valid_labels: list) -> str:
    """Fast regex/keyword fallback for binary classification tasks.

    Tiered matching strategy (returns on first successful tier):
      1. Explicit ``Answer: (A) label`` or ``Answer: label`` pattern
      2. Inline choice pattern ``(A) label`` / ``(B) label`` — last occurrence wins
      3. Last occurrence of any valid label keyword in the text
    Returns 'unknown' if nothing matches.
    """
    valid_set = {l.lower() for l in valid_labels}
    lower = output.lower()

    # Tier 1: explicit "Answer:" prefix followed by letter+label or just label
    m = re.search(r"answer\s*[:\s]*\(?[a-z]\)?\s*([a-z ]+)", lower)
    if m:
        candidate = m.group(1).strip().split()[0]
        if candidate in valid_set:
            return candidate

    # Tier 2: inline "(A) label" / "(B) label" choice pattern — use last match
    # so the conclusion overrides any earlier mention of options
    matches = list(re.finditer(r"\([a-z]\)\s*([a-z]+)", lower))
    for m in reversed(matches):
        candidate = m.group(1).strip()
        if candidate in valid_set:
            return candidate

    # Tier 3: last occurrence of any valid label in the full text
    last_pos = -1
    last_label = None
    for label in valid_labels:
        pos = lower.rfind(label.lower())
        if pos > last_pos:
            last_pos = pos
            last_label = label.lower()
    if last_label is not None:
        return last_label

    return 'unknown'


def extract_answer_from_output(model, tokenizer, output, valid_labels=None):
    """
    Extracts the most likely answer from the output string using an LLM.

    Args:
        model: language model for extraction
        tokenizer: tokenizer for the model
        output: raw model output string to extract answer from
        valid_labels: list of valid label strings to choose from.
                      Defaults to ['cross', 'yield'] for backward compatibility.
    """
    if valid_labels is None:
        valid_labels = ['cross', 'yield']
    labels_str = ' or '.join(f"'{l}'" for l in valid_labels)
    valid_set = {l.lower() for l in valid_labels}

    prompt = (f"Based on the description, find out which of these is the most likely answer: "
              f"{labels_str}. Respond with exactly one of those options.\nDescription:\n{output}")

    try:
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. You must output exactly one of: {labels_str}."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=20
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip().lower()
        if answer in valid_set:
            return answer
        # Try partial match in case model outputs a substring
        for label in valid_labels:
            if label.lower() in answer:
                return label.lower()
        return answer
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "unknown"

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m utils.parser.extract_answers <path_to_responses.json>")
        sys.exit(1)
    filepath = sys.argv[1]
    output_filepath = filepath.replace("_responses.json", "_responses_fixed.json")

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check if there are any answers to fix before loading the model
    needs_update = False
    for item in data:
        if item.get('pred_answer', '') not in ['cross', 'yield']:
            needs_update = True
            break
            
    updated_count, unfinished_count = 0, 0
    if needs_update:
        print("Loading local Qwen model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        infer_time = 0
        
        for item in data:
            out = item.get('out', '')
            reasoning = item.get('reasoning', '')
            current_pred = item.get('pred_answer', '')
            
            # Only overwrite if pred_answer is incomplete or not one of the two choices
            is_finished = item.get('is_finished') if item.get('is_finished') else reasoning.rstrip().endswith('.')
            if is_finished:
                if current_pred not in ['cross', 'yield']:
                    match = re.search(r"^Answer?\s*[<(]?([A-Z])?[>)]?\s*(.*)", item['out'], re.MULTILINE)
                    if match and match.group(2).strip().lower() in ['cross', 'yield']:
                        item['pred_answer'] = match.group(2)
                    else:
                        match = re.search(r"Answer\s*[<(]?([A-Z])?[>)]", item['out'])
                        if match:
                            item['pred_answer'] = item['options'][match.group(1)]
                        else:
                            extracted = extract_answer_from_output(model, tokenizer, out)
                            item['pred_answer'] = extracted
                    updated_count += 1
            else:
                item['pred_answer'] = 'unknown'
                unfinished_count += 1
            item['is_finished'] = is_finished
            if not reasoning: 
                item['reasoning'] = out
            infer_time += item.get('duration', 0)
        avg_infer_time = infer_time / len(data) if data else 0
        
    else:
        print("No missing answers found!")
            
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Processed {len(data)} items. \nFixed {updated_count} answers.\nMarked {unfinished_count} items as unfinished.")
    print(f"Saved corrected file to {output_filepath}")

    report = analyze(output_filepath)
    print(f"Average inference time: {avg_infer_time:.2f} seconds")
    print('Accuracy:{:.3f}'.format(report['classification']['accuracy']))
    print('MacroF1 fair:{:.3f}'.format((report['classification']['per_class']['cross']['f1'] + report['classification']['per_class']['yield']['f1'])/2))
    pretty_print_report(report)


if __name__ == "__main__":
    main()
