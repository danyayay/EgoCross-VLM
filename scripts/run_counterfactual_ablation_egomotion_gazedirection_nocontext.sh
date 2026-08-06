#!/usr/bin/env bash
# Compare base and final LoRA models under post-tuning visual/temporal controls,
# repeated across the 3 training seeds used to assess robustness.
#
# Required:
#   ADAPTER_PATHS="logs/.../seed_42/.../best-... logs/.../seed_43/.../best-... logs/.../seed_44/.../best-..." \
#       bash scripts/run_counterfactual_ablation_egomotion_gazedirection.sh
#
# LOG_DIR is derived per adapter from its own last path component (typically a
# training-run timestamp), so each seed's ablation results land in their own
# subdirectory under LOG_DIR_ROOT. Override LOG_DIR directly to disable this.
#
# Useful overrides:
#   SAMPLE_N=20 DRY_RUN=1 bash scripts/run_counterfactual_ablation_egomotion_gazedirection.sh

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
ADAPTER_PATHS="${ADAPTER_PATHS:-\
logs/vlm_training_dot/seed_42/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260529_000601_ctxego_motion+gaze_direction_cfps4_preface_legacy \
logs/vlm_training_dot/seed_43/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260717_223227_ctxego_motion+gaze_direction_cfps4_preface_legacy \
logs/vlm_training_dot/seed_44/cotnone_f4_interleaved_p6_ctxego_motion+gaze_direction_cfps4_preface_legacy__lora_vlm_bridger_r2_a8_lr1e-4/Qwen3-VL-2B-Instruct/20260719_000202_ctxego_motion+gaze_direction_cfps4_preface_legacy}"
ANN_FILE="${ANN_FILE:-features/groundvqa/annotations.VRbinary__crossing_intention__test_close.json}"
VIDEO_ROOT="${VIDEO_ROOT:-data/videodata_256/clips_dot}"
LOG_DIR_ROOT="${LOG_DIR_ROOT:-logs/counterfactual_ablation_egomotion_gaze_direction_nocontext}"
CACHE_DIR="${CACHE_DIR:-.cache}"
NUM_FRAMES="${NUM_FRAMES:-4}"
TCOT_TYPE="${TCOT_TYPE:-none}"
PROMPT_VARIANT="${PROMPT_VARIANT:-p6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_N="${SAMPLE_N:-}"
DRY_RUN="${DRY_RUN:-0}"
SHUFFLE_SEEDS="${SHUFFLE_SEEDS:-42 43 44}"

CONTEXT_FEATURE_SETS="${CONTEXT_FEATURE_SETS:-none}"

if [[ -z "$ADAPTER_PATHS" ]]; then
    echo "ADAPTER_PATHS must list the validation-selected LoRA adapter for each training seed." >&2
    exit 2
fi
if [[ ! -f "$ANN_FILE" ]]; then
    echo "Annotation file not found: $ANN_FILE" >&2
    exit 2
fi
if [[ ! -d "$VIDEO_ROOT" ]]; then
    echo "Video root not found: $VIDEO_ROOT" >&2
    exit 2
fi

run_eval() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

for adapter_path in $ADAPTER_PATHS; do
    if [[ ! -d "$adapter_path" ]]; then
        echo "Adapter path not found: $adapter_path" >&2
        exit 2
    fi

    # Tag the log dir with the adapter's own run identifier (its last path
    # component, typically a training-run timestamp) so seeds don't collide.
    adapter_tag="$(basename "$adapter_path")"
    log_dir="${LOG_DIR:-$LOG_DIR_ROOT/$adapter_tag}"

    echo ""
    echo "======================================================================="
    echo "  adapter: $adapter_path"
    echo "  log_dir: $log_dir"
    echo "======================================================================="

    common=(
        python -m training.eval_vlm
        --model_name "$MODEL_NAME"
        --ann_file "$ANN_FILE"
        --video_root "$VIDEO_ROOT"
        --log_dir "$log_dir"
        --cache_dir "$CACHE_DIR"
        --num_frames "$NUM_FRAMES"
        --tcot_type "$TCOT_TYPE"
        --prompt_variant "$PROMPT_VARIANT"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --context_features "$CONTEXT_FEATURE_SETS"
        --interleaved_timestamps
    )

    if [[ -n "$SAMPLE_N" ]]; then
        common+=(--sample_n "$SAMPLE_N" --sample_seed "$SAMPLE_SEED")
    fi

    for model_variant in finetuned; do
        model_args=()
        if [[ "$model_variant" == "finetuned" ]]; then
            model_args=(--lora_adapter "$adapter_path")
        fi

        # Original and text-only controls use the same prompt/decoding settings.
        run_eval "${common[@]}" "${model_args[@]}" --visual_ablation original
        run_eval "${common[@]}" "${model_args[@]}" --no_video

        for condition in black mismatched reverse; do
            run_eval "${common[@]}" "${model_args[@]}" \
                --visual_ablation "$condition" --ablation_seed "$SAMPLE_SEED"
        done

        for shuffle_seed in $SHUFFLE_SEEDS; do
            run_eval "${common[@]}" "${model_args[@]}" \
                --visual_ablation shuffle --ablation_seed "$shuffle_seed"
            
            run_eval "${common[@]}" "${model_args[@]}" \
                --visual_ablation noise --ablation_seed "$shuffle_seed"
        done
    done
done



TOPIC="aicheetah_ntfy_"

curl -d "✅ Done!" \
    -H "Counterfactual + context guidance + no context evaluation" \
    -H "Priority: high" \
    -H "Tags: white_check_mark,computer" \
    ntfy.sh/$TOPIC