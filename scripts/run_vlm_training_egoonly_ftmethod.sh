#!/usr/bin/env bash
# Train VLM LoRA adapters from scratch across prompt/frame settings.
#
# Run from repo root, for example:
#   conda run -n pedintention bash scripts/run_vlm_training_effect_evals2.sh
#
# Common overrides:
#   ANN_TEMPLATES="features/groundvqa/annotations.VRbinary__crossing_intention__mode_close.json" \
#     NUM_FRAMES_SET="4 8 16" COT_TYPES="none tcot4" DRY_RUN=1 \
#     bash scripts/run_vlm_training_effect_evals2.sh
#   CONTEXT_FEATURE_SETS="none ego_motion gaze_direction ego_motion,gaze_direction" \
#     CONTEXT_FEATURE_FPS_SET="auto 8" DRY_RUN=1 \
#     bash scripts/run_vlm_training_effect_evals2.sh
#   FT_TYPES="lora_llm_vlm_bridger lora_vlm_bridger" LORA_RANKS="2 4" LORA_ALPHAS="8 16" \
#     bash scripts/run_vlm_training_effect_evals2.sh
#   SAMPLE_N=100 SAMPLE_SEED=42 bash scripts/run_vlm_training_effect_evals2.sh

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
ANN_TEMPLATES="${ANN_TEMPLATES:-features/groundvqa/annotations.VRbinary__crossing_intention__mode_close.json}"
VIDEO_ROOT="${VIDEO_ROOT:-data/videodata_256/clips}"
LOG_DIR="${LOG_DIR:-logs/vlm_training}"
CACHE_DIR="${CACHE_DIR:-.cache}"

NUM_FRAMES_SET="${NUM_FRAMES_SET:-8}"
COT_TYPES="${COT_TYPES:-none}"
PROMPT_VARIANTS="${PROMPT_VARIANTS:-p6}"
INTERLEAVED_SET="${INTERLEAVED_SET:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEEDS="${SEEDS:-43 44}"
SAMPLE_N="${SAMPLE_N:-}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
TASK_TYPE="${TASK_TYPE:-crossing_intention}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-1}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
TOP_K="${TOP_K:-1}"
MONITOR="${MONITOR:-val_acc}"
# FT_TYPES="${FT_TYPES:-lora_llm_attn_qkvo lora_llm_attn_mlp lora_llm_vlm_bridger lora_vlm_bridger}"
FT_TYPES="${FT_TYPES:-lora_vlm_bridger lora_llm_attn_mlp lora_llm_vlm_bridger}"
LORA_RANKS="${LORA_RANKS:-2}"
LORA_ALPHAS="${LORA_ALPHAS:-8}"

DRY_RUN="${DRY_RUN:-0}"

# shellcheck disable=SC2206
ann_templates=($ANN_TEMPLATES)
# shellcheck disable=SC2206
num_frames_values=($NUM_FRAMES_SET)
# shellcheck disable=SC2206
cot_types=($COT_TYPES)
# shellcheck disable=SC2206
prompt_variants=($PROMPT_VARIANTS)
# shellcheck disable=SC2206
interleaved_values=($INTERLEAVED_SET)
# shellcheck disable=SC2206
ft_types=($FT_TYPES)
# shellcheck disable=SC2206
lora_ranks=($LORA_RANKS)
# shellcheck disable=SC2206
lora_alphas=($LORA_ALPHAS)

echo "======================================================================="
echo "  Fresh VLM LoRA training"
echo "======================================================================="
echo "model:        $MODEL_NAME"
echo "ann_templates:$ANN_TEMPLATES"
echo "video_root:   $VIDEO_ROOT"
echo "frames:       $NUM_FRAMES_SET"
echo "cot_types:    $COT_TYPES"
echo "prompts:      $PROMPT_VARIANTS"
echo "interleaved:  $INTERLEAVED_SET"
echo "ft_types:     $FT_TYPES"
echo "lora_ranks:   $LORA_RANKS"
echo "lora_alphas:  $LORA_ALPHAS"
echo "epochs:       $EPOCHS"
echo "batch_size:   $BATCH_SIZE"
echo "lr:           $LR"
echo "top_k:        $TOP_K"
echo "monitor:      $MONITOR"
echo "max_tokens:   $MAX_NEW_TOKENS"
if [[ -n "$SAMPLE_N" ]]; then
    echo "sample_n:     $SAMPLE_N"
    echo "sample_seed:  $SAMPLE_SEED"
    LOG_DIR="${LOG_DIR}_debugging"
fi
echo "log_dir:      $LOG_DIR"
echo "dry_run:      $DRY_RUN"

FAILED_RUNS=0

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@"
        echo ""
        return 0
    fi

    set +e
    "$@"
    local status=$?
    set -e

    if [[ "$status" -eq 0 ]]; then
        return 0
    fi

    FAILED_RUNS=$((FAILED_RUNS + 1))
    echo ""
    echo "WARNING: training command failed with exit code $status; continuing to next run."
    return 0
}

ann_tag() {
    local template="$1"
    local name
    name="${template##*/}"
    name="${name#annotations.VRbinary_}"
    name="${name%.json}"
    name="${name//_mode_close/}"
    name="${name//__/_}"
    name="${name//[^A-Za-z0-9._-]/_}"
    echo "$name"
}

for ann_template in "${ann_templates[@]}"; do
    ann_train="${ann_template/mode/train}"
    ann_val="${ann_template/mode/val}"
    ann_test="${ann_template/mode/test}"
    for split_file in "$ann_train" "$ann_val" "$ann_test"; do
        if [[ ! -f "$split_file" ]]; then
            echo "Missing annotation split derived from template: $split_file"
            exit 1
        fi
    done

    for SEED in $SEEDS; do
        for num_frames in "${num_frames_values[@]}"; do
            for cot_type in "${cot_types[@]}"; do
                for prompt_variant in "${prompt_variants[@]}"; do
                    for interleaved in "${interleaved_values[@]}"; do
                        interleaved_tag="nointerleave"
                        interleaved_args=()
                        if [[ "$interleaved" == "1" ]]; then
                            interleaved_tag="interleaved"
                            interleaved_args=(--interleaved_timestamps)
                        fi
                        for ft_type in "${ft_types[@]}"; do
                            for lora_rank in "${lora_ranks[@]}"; do
                                for lora_alpha in "${lora_alphas[@]}"; do
                                    run_tag="cot${cot_type}_f${num_frames}_${interleaved_tag}_${prompt_variant}"
                                    run_tag+="__${ft_type}_r${lora_rank}_a${lora_alpha}_lr${LR}"
                                    train_log_dir="$LOG_DIR/seed_$SEED/$run_tag"
                                    cmd=(
                                        python -m training.train_vlm
                                        --mode train
                                        --model_name "$MODEL_NAME"
                                        --ann_file_template "$ann_template"
                                        --video_root "$VIDEO_ROOT"
                                        --cache_dir "$CACHE_DIR"
                                        --log_dir "$train_log_dir"
                                        --init_adapter_path ""
                                        --num_frames "$num_frames"
                                        --cot_type "$cot_type"
                                        --prompt_variant "$prompt_variant"
                                        --task_type "$TASK_TYPE"
                                        --eval_max_new_tokens "$MAX_NEW_TOKENS"
                                        --random_seed "$SEED"
                                        --epochs "$EPOCHS"
                                        --batch_size "$BATCH_SIZE"
                                        --lr "$LR"
                                        --top_k "$TOP_K"
                                        --monitor "$MONITOR"
                                        --ft_type "$ft_type"
                                        --lora_rank "$lora_rank"
                                        --lora_alpha "$lora_alpha"
                                    )
                                    cmd+=("${interleaved_args[@]}")
                                    if [[ "$EVAL_DETERMINISTIC" == "1" ]]; then
                                        cmd+=(--eval_deterministic)
                                    fi
                                    if [[ -n "$SAMPLE_N" ]]; then
                                        cmd+=(--sample_n "$SAMPLE_N" --sample_seed "$SAMPLE_SEED")
                                    fi

                                    echo ""
                                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                                    echo "  ann_template=$ann_template"
                                    echo "  run_tag=$run_tag"
                                    echo "  log_root=$train_log_dir"
                                    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                                    run_cmd "${cmd[@]}"
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo ""
if [[ "$FAILED_RUNS" -gt 0 ]]; then
    echo "Done fresh VLM LoRA training with $FAILED_RUNS failed run(s)."
else
    echo "Done fresh VLM LoRA training."
fi
