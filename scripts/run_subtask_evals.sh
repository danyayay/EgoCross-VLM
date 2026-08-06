#!/bin/bash
# Run zero-shot evaluations for all subtasks (no CoT, then cot4)
# Usage: conda run -n work3 bash run_subtask_evals.sh

# set -e
# TASKS="speed approach ped_moving ped_direction gaze_vehicle ehmi vehicle_proximity crossing_proximity head_turning"
# ANN_DIR="features/groundvqa_auxtask"
# VIDEO_ROOT="data/videodata_256/clips"
# MODEL="Qwen/Qwen3-VL-2B-Instruct"
# LOG_BASE="logs/subtask_eval"

# echo "=== Phase 1: No-CoT (plain) evaluations ==="
# for task in $TASKS; do
#     ANN_FILE="${ANN_DIR}/annotations.VRbinary__aux_${task}__test_close.json"
#     echo "--- Task: ${task} | cot: None ---"
#     python -m training.eval_qwen \
#         --ann_file "${ANN_FILE}" \
#         --video_root "${VIDEO_ROOT}" \
#         --model_name "${MODEL}" \
#         --cot_type none \
#         --log_dir "${LOG_BASE}_None" \
#         --seed 42
# done

# echo "=== Phase 2: cot4 evaluations ==="
# for task in $TASKS; do
#     ANN_FILE="${ANN_DIR}/annotations.VRbinary__aux_${task}__test_close.json"
#     echo "--- Task: ${task} | cot: cot4 ---"
#     python -m training.eval_qwen \
#         --ann_file "${ANN_FILE}" \
#         --video_root "${VIDEO_ROOT}" \
#         --model_name "${MODEL}" \
#         --cot_type cot4 \
#         --log_dir "${LOG_BASE}_cot4" \
#         --seed 42
# done

# echo "=== Phase 3: cot7 on main crossing intention task ==="
# python -m training.eval_qwen \
#     --ann_file "features/groundvqa_qn0/annotations.VRbinary__crossing_intention__test_close.json" \
#     --video_root "${VIDEO_ROOT}" \
#     --model_name "${MODEL}" \
#     --cot_type cot7 \
#     --log_dir "logs/qwen_new_qn3_cot7_eval" \
#     --seed 42

# echo "=== All evaluations done ==="

TOPIC="aicheetah_ntfy_"


MODEL="Qwen/Qwen3-VL-2B-Instruct"

python -m training.eval_auxtasks \
    --model_name "${MODEL}" \

python -m training.eval_auxtasks \
    --model_name "${MODEL}" \
    --interleaved_timestamps




MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
python -m training.eval_auxtasks \
    --model_name "${MODEL}" \
    --quantize

python -m training.eval_auxtasks \
    --model_name "${MODEL}" \
    --quantize \
    --interleaved_timestamps


curl -d "✅ Eval auxtasks is done!" \
    -H "Title: Task Complete" \
    -H "Priority: high" \
    -H "Tags: white_check_mark,computer" \
    ntfy.sh/$TOPIC