#!/usr/bin/env bash
# Compare base Qwen3-VL model(s) against LoRA fine-tuned adapter(s) on MMBench,
# across multiple seeds, to check whether fine-tuning on the pedestrian-intention
# task degrades general multimodal ability.
#
# Run from repo root, for example:
#   conda run -n work3 bash scripts/run_mmbench_eval.sh \
#       LORA_ADAPTERS="logs/qwen_training_intention/Qwen3-VL-2B-Instruct/20260415_085039/best-acc-0.81-step-500"
#
#   SUBSET_SEEDS="42 43 44" SAMPLES_PER_CATEGORY=100 CATEGORY_LEVEL=l2-category \
# Common overrides:
#     bash scripts/run_mmbench_eval.sh
#   LORA_ADAPTERS="adapter/path/one adapter/path/two" bash scripts/run_mmbench_eval.sh
#   DRY_RUN=1 bash scripts/run_mmbench_eval.sh

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
# LORA_ADAPTERS="${LORA_ADAPTERS:-\
# logs/vlm_training/seed_42/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260528_034756 \
# logs/vlm_training/seed_43/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260723_234732 \
# logs/vlm_training/seed_44/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260724_162218 \
# logs/vlm_training_dot/seed_42/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260529_000601_ctxego_motion+gaze_direction_cfps4_preface_legacy \
# logs/vlm_training_dot/seed_43/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260717_223227_ctxego_motion+gaze_direction_cfps4_preface_legacy \
# logs/vlm_training_dot/seed_44/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260719_000202_ctxego_motion+gaze_direction_cfps4_preface_legacy}"
MODEL1_NAME="${MODEL1_NAME:-Ego-only}"
MODEL1_ADAPTERS="${MODEL1_ADAPTERS:-\
logs/vlm_training/seed_42/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260528_034756 \
logs/vlm_training/seed_43/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260723_234732 \
logs/vlm_training/seed_44/cotnone_f8_interleaved_p6__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260724_162218}"
MODEL2_NAME="${MODEL2_NAME:-Ego+Gaze}"
MODEL2_ADAPTERS="${MODEL2_ADAPTERS:-\
logs/vlm_training_dot/seed_42/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260529_000601_ctxego_motion+gaze_direction_cfps4_preface_legacy \
logs/vlm_training_dot/seed_43/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260717_223227_ctxego_motion+gaze_direction_cfps4_preface_legacy \
logs/vlm_training_dot/seed_44/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260719_000202_ctxego_motion+gaze_direction_cfps4_preface_legacy}"
LORA_ADAPTERS="${LORA_ADAPTERS:-$MODEL1_ADAPTERS $MODEL2_ADAPTERS}"
CATEGORY_LEVEL="${CATEGORY_LEVEL:-l2-category}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-100}"
SUBSET_SEEDS="${SUBSET_SEEDS:-42 43 44}"
LOG_DIR="${LOG_DIR:-logs/mmbench_eval}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-results/mmbench_compare}"
GENERALIZATION_OUT_DIR="${GENERALIZATION_OUT_DIR:-results/mmbench_generalization}"
DROP_THRESHOLD="${DROP_THRESHOLD:-0.05}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DRY_RUN="${DRY_RUN:-0}"

model_short="${MODEL_NAME##*/}"

run() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf '%q ' "$@"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

declare -A base_report_by_seed

for SUBSET_SEED in $SUBSET_SEEDS; do
    sampled_indices_file=".cache/mmbench_sampled_indices_${CATEGORY_LEVEL}_${SAMPLES_PER_CATEGORY}_subset-seed-${SUBSET_SEED}.json"
    run_log_dir="${LOG_DIR}"
    
    echo ""
    echo "=== Subset Seed $SUBSET_SEED: evaluating base model ($MODEL_NAME) on MMBench... ==="
    run python -m training.eval_mmbench \
        --model_name "$MODEL_NAME" \
        --category_level "$CATEGORY_LEVEL" \
        --samples_per_category "$SAMPLES_PER_CATEGORY" \
        --sampled_indices_file "$sampled_indices_file" \
        --log_dir "$run_log_dir" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --subset_seed "$SUBSET_SEED"

    if [[ "$DRY_RUN" != "1" ]]; then
        base_report_by_seed["$SUBSET_SEED"]=$(ls -t "${run_log_dir}/${model_short}/${CATEGORY_LEVEL}_${SAMPLES_PER_CATEGORY}_subset-seed-${SUBSET_SEED}"/base_*_report.json | head -n1)
    fi

    for LORA_ADAPTER in $LORA_ADAPTERS; do
        echo "=== Subset Seed $SUBSET_SEED: evaluating adapter $LORA_ADAPTER on MMBench... ==="
        run python -m training.eval_mmbench \
            --model_name "$MODEL_NAME" \
            --lora_adapter "$LORA_ADAPTER" \
            --category_level "$CATEGORY_LEVEL" \
            --samples_per_category "$SAMPLES_PER_CATEGORY" \
            --sampled_indices_file "$sampled_indices_file" \
            --log_dir "$run_log_dir" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --subset_seed "$SUBSET_SEED"
    done
done

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "Dry run: skipping aggregation/comparison step (report paths depend on actual eval runs)."
    exit 0
fi

echo ""
echo "=== Aggregating base-vs-finetuned MMBench accuracy across $(echo "$SUBSET_SEEDS" | wc -w) seed(s)... ==="
for LORA_ADAPTER in $LORA_ADAPTERS; do
    adapter_name="$(basename "$LORA_ADAPTER")"

    if [[ "$LORA_ADAPTER" =~ seed_([0-9]+) ]]; then
        model_seed="model-seed-${BASH_REMATCH[1]}"
    else
        model_seed="$(basename "$LORA_ADAPTER")"
    fi

    base_reports=()
    finetuned_reports=()
    for SUBSET_SEED in $SUBSET_SEEDS; do
        run_log_dir="${LOG_DIR}"
        base_reports+=("${base_report_by_seed[$SUBSET_SEED]}")
        finetuned_reports+=("${run_log_dir}/${model_short}/${CATEGORY_LEVEL}_${SAMPLES_PER_CATEGORY}_subset-seed-${SUBSET_SEED}/${model_seed}/${adapter_name}_report.json")
    done

    echo ""
    echo "--- Adapter: $adapter_name ---"
    run python -m utils.analyze_mmbench_results \
        --base_reports "${base_reports[@]}" \
        --finetuned_reports "${finetuned_reports[@]}" \
        --out_dir "${COMPARE_OUT_DIR}/${adapter_name}" \
        --drop_threshold "$DROP_THRESHOLD" \
        --plot
done

echo ""
echo "=== Generalization-robustness summary: Base vs. $MODEL1_NAME vs. $MODEL2_NAME ==="
run python -m utils.analyze_mmbench_results generalization \
    --log_dir "$LOG_DIR" \
    --model_name "$MODEL_NAME" \
    --category_level "$CATEGORY_LEVEL" \
    --samples_per_category "$SAMPLES_PER_CATEGORY" \
    --subset_seeds $SUBSET_SEEDS \
    --model1_name "$MODEL1_NAME" \
    --model1_adapters $MODEL1_ADAPTERS \
    --model2_name "$MODEL2_NAME" \
    --model2_adapters $MODEL2_ADAPTERS \
    --out_dir "$GENERALIZATION_OUT_DIR" \
    --plot

echo ""
echo "Done MMBench regression check."
