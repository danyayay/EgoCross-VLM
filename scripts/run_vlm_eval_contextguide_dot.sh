#!/usr/bin/env bash
# Evaluate VLM performance across contextual-input prompt variants from one structured annotation JSON.
#
# Run from repo root, for example:
#   conda run -n pedintention bash scripts/run_contextual_input_evals.sh
#
# Common overrides:
#   ANN_DIR=features/groundvqa_qn0 CONTEXT_SETUPS="00000 00010 00100 00001" \
#     bash scripts/run_contextual_input_evals.sh
#   CONTEXT_SETUPS="00000 00010_every3 00100_every3" SAMPLE_N=100 \
#     bash scripts/run_contextual_input_evals.sh
#   DRY_RUN=1 bash scripts/run_contextual_input_evals.sh

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
ANN_FILE="${ANN_FILE:-features/groundvqa/annotations.VRbinary__crossing_intention__test_close.json}"
VIDEO_ROOT="${VIDEO_ROOT:-data/videodata_256/clips_dot}"
LOG_DIR="${LOG_DIR:-logs/vlm_eval_context_guided_dot}"
CACHE_DIR="${CACHE_DIR:-.cache}"
NUM_FRAMES="${NUM_FRAMES:-8}"
TCOT_TYPE="${TCOT_TYPE:-none}"
PROMPT_VARIANT="${PROMPT_VARIANT:-p6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEEDS="${SEEDS:-43 44}"
INTERLEAVED="${INTERLEAVED:-1}"
QUANTIZE="${QUANTIZE:-1}"
SAMPLE_N="${SAMPLE_N:-}"
SAMPLE_SEED="${SAMPLE_SEED:-}"
DRY_RUN="${DRY_RUN:-0}"
# CONTEXT_FEATURE_SETS="${CONTEXT_FEATURE_SETS:-ego_motion gaze_direction gaze_on_screen_ratio vehicle_motion ego_motion,gaze_direction ego_motion,gaze_on_screen_ratio ego_motion,vehicle_motion}"
# CONTEXT_FEATURE_SETS="${CONTEXT_FEATURE_SETS:-ego_motion gaze_direction gaze_direction_change gaze_on_screen_ratio vehicle_motion ego_motion,gaze_direction ego_motion,gaze_direction_change ego_motion,gaze_on_screen_ratio ego_motion,vehicle_motion}"
# CONTEXT_FEATURE_SETS="${CONTEXT_FEATURE_SETS:-ego_motion,gaze_direction ego_motion,gaze_direction_change ego_motion,gaze_on_screen_ratio ego_motion,vehicle_motion}"
CONTEXT_FEATURE_SETS="${CONTEXT_FEATURE_SETS:-none}"
# TODO: for demographics, trust/behavior scores should be formated as score/full score.
# TODO: update the compact: "compact clip cues" to feature-dependent.
CONTEXT_FEATURE_FPS_SET="${CONTEXT_FEATURE_FPS_SET:-auto}"
CONTEXT_PROMPT_MODES="${CONTEXT_PROMPT_MODES:-preface}"
# CONTEXT_PROMPT_MODES="${CONTEXT_PROMPT_MODES:-preface}"
CONTEXT_FEATURE_FORMATS="${CONTEXT_FEATURE_FORMATS:-legacy}"
# CONTEXT_FEATURE_FORMATS="${CONTEXT_FEATURE_FORMATS:-compact}"
# CONTEXT_FEATURE_INTERPRETATIONS="${CONTEXT_FEATURE_INTERPRETATIONS:-none brief detailed}"
CONTEXT_FEATURE_INTERPRETATIONS="${CONTEXT_FEATURE_INTERPRETATIONS:-none}"

model_short="${MODEL_NAME##*/}"
interleaved_tag="nointerleave"
if [[ "$INTERLEAVED" == "1" ]]; then
    interleaved_tag="interleaved"
fi

base_run_tag="tcot${TCOT_TYPE}_f${NUM_FRAMES}_${interleaved_tag}_${PROMPT_VARIANT}"

echo "======================================================================="
echo "  Contextual input evaluation"
echo "======================================================================="
echo "model:       $MODEL_NAME"
echo "ann_file:    $ANN_FILE"
echo "features:    $CONTEXT_FEATURE_SETS"
echo "feature_fps: $CONTEXT_FEATURE_FPS_SET"
echo "ctx_modes:   $CONTEXT_PROMPT_MODES"
echo "ctx_formats: $CONTEXT_FEATURE_FORMATS"
echo "ctx_interp:  $CONTEXT_FEATURE_INTERPRETATIONS"
echo "frames:      $NUM_FRAMES"
echo "tcot:        $TCOT_TYPE"
echo "prompt:      $PROMPT_VARIANT"
echo "interleaved: $INTERLEAVED"
echo "quantize:    $QUANTIZE"
echo "log_dir:     $LOG_DIR"
echo "seeds:       $SEEDS"
echo "dry_run:     $DRY_RUN"
if [[ -n "$SAMPLE_N" ]]; then
    echo "sample_n:    $SAMPLE_N"
fi

if [[ ! -f "$ANN_FILE" ]]; then
    echo "Missing annotation file: $ANN_FILE"
    exit 1
fi

# shellcheck disable=SC2206
feature_sets=($CONTEXT_FEATURE_SETS)
# shellcheck disable=SC2206
feature_fps_values=($CONTEXT_FEATURE_FPS_SET)
# shellcheck disable=SC2206
prompt_modes=($CONTEXT_PROMPT_MODES)
# shellcheck disable=SC2206
feature_formats=($CONTEXT_FEATURE_FORMATS)
# shellcheck disable=SC2206
feature_interpretations=($CONTEXT_FEATURE_INTERPRETATIONS)
first_context_feature_fps="${feature_fps_values[0]}"
first_context_prompt_mode="${prompt_modes[0]}"
first_context_feature_format="${feature_formats[0]}"
first_context_feature_interpretation="${feature_interpretations[0]}"

for SEED in $SEEDS; do
  seed_sample_seed="${SAMPLE_SEED:-$SEED}"
  for context_features in "${feature_sets[@]}"; do
    for context_feature_fps in "${feature_fps_values[@]}"; do
      for context_prompt_mode in "${prompt_modes[@]}"; do
        for context_feature_format in "${feature_formats[@]}"; do
          for context_feature_interpretation in "${feature_interpretations[@]}"; do
            if [[ "$context_features" == "none" && ( "$context_feature_fps" != "$first_context_feature_fps" || "$context_prompt_mode" != "$first_context_prompt_mode" || "$context_feature_format" != "$first_context_feature_format" || "$context_feature_interpretation" != "$first_context_feature_interpretation" ) ]]; then
                continue
            fi
            feature_tag="${context_features//,/+}"
            run_tag="${base_run_tag}"
            cmd=(
                python -m training.eval_vlm
                --model_name "$MODEL_NAME"
                --ann_file "$ANN_FILE"
                --video_root "$VIDEO_ROOT"
                --log_dir "$LOG_DIR"
                --cache_dir "$CACHE_DIR"
                --num_frames "$NUM_FRAMES"
                --tcot_type "$TCOT_TYPE"
                --prompt_variant "$PROMPT_VARIANT"
                --context_features "$context_features"
                --max_new_tokens "$MAX_NEW_TOKENS"
                --seed "$SEED"
            )
            if [[ "$context_features" != "none" ]]; then
                run_tag+="_ctx${feature_tag}_cfps${context_feature_fps}_${context_prompt_mode}_${context_feature_format}"
                cmd+=(
                    --context_feature_fps "$context_feature_fps"
                    --context_prompt_mode "$context_prompt_mode"
                    --context_feature_format "$context_feature_format"
                    --context_feature_interpretation "$context_feature_interpretation"
                )
                if [[ "$context_feature_interpretation" != "none" ]]; then
                    run_tag+="_interp${context_feature_interpretation}"
                fi
            fi

            if [[ "$INTERLEAVED" == "1" ]]; then
                cmd+=(--interleaved_timestamps)
            fi
            if [[ "$QUANTIZE" == "1" ]]; then
                run_tag+="_int8"
                cmd+=(--quantize)
            fi
            if [[ -n "$SAMPLE_N" ]]; then
                cmd+=(--sample_n "$SAMPLE_N" --sample_seed "$seed_sample_seed")
            fi

            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  seed=$SEED"
            echo "  context_features=$context_features"
            echo "  context_feature_fps=$context_feature_fps"
            echo "  context_prompt_mode=$context_prompt_mode"
            echo "  context_feature_format=$context_feature_format"
            echo "  context_feature_interpretation=$context_feature_interpretation"
            echo "  ann_file=$ANN_FILE"
            echo "  log_root=$LOG_DIR"
            echo "  expected_run_tag=$run_tag"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            if [[ "$DRY_RUN" == "1" ]]; then
                printf '%q ' "${cmd[@]}"
                echo ""
            else
                "${cmd[@]}"
            fi
          done
        done
      done
    done
  done
done
echo ""
echo "Done contextual input evaluation."




TOPIC="aicheetah_ntfy_"

curl -d "✅ $STEP is done!" \
    -H "Title: Zero-shot + Context guidance" \
    -H "Priority: high" \
    -H "Tags: white_check_mark,computer" \
    ntfy.sh/$TOPIC