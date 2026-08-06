#!/usr/bin/env bash
# run_vlm_eval_grid.sh — Zero-shot VLM evaluation experiment sweep
#
# Run from the project root with the pedintention conda environment active:
#   conda run -n pedintention bash scripts/run_vlm_eval_grid.sh [STEP]
#
# STEP controls which experiment table is run:
#   frame_ablation   (Table 2) — frame count & interleaving ablation on Qwen3-VL-2B
#   model_comparison (Table 1) — all models at best_N frames
#   cot_ablation     (Table 3) — text-CoT prompt variants on Qwen3-VL-2B
#   contextual_inputs — contextual annotation JSON ablation, e.g. 00000/00010/00100
#   all              — run all three in sequence (default)
#
# Override the default frame count for Tables 1 & 3 with:
#   BEST_N=16 bash scripts/run_vlm_eval_grid.sh model_comparison 

set -euo pipefail

STEP="${1:-cot_visual_runs}"  # default step to run (frame_ablation, model_comparison, cot_ablation, contextual_inputs, all)
BEST_N="${BEST_N:-8}"           # default: 16 frames (update after Table 2 results)
PROMPT_VARIANT="${PROMPT_VARIANT:-p6}"   # default prompt variant (p0–p6); override e.g. PROMPT_VARIANT=p4
TCOT="${TCOT:-none}"             # default CoT type (none, tcot1, tcot4, tcot7); override e.g. TCOT=tcot4
ANN_FILE="features/groundvqa/annotations.VRbinary__crossing_intention__test_close.json"
VIDEO_ROOT="data/videodata_256/clips"
LOG_DIR="logs/vlm_eval"
CACHE_DIR=".cache"
MAX_NEW_TOKENS=1024
SEEDS=(43 44)

# ── helpers ────────────────────────────────────────────────────────────────────────
run() {
    # run MODEL NUM_FRAMES COT INTERLEAVED [EXTRA_ARGS...]
    # Respects $PROMPT_VARIANT (default p6); override per-call via extra args e.g. --prompt_variant p4
    local model="$1" num_frames="$2" tcot="$3" interleaved="$4"
    shift 4
    local extra=("$@")

    local cmd=(
        python -m training.eval_vlm
        --model_name "$model"
        --ann_file "$ANN_FILE"
        --video_root "$VIDEO_ROOT"
        --log_dir "$LOG_DIR"
        --cache_dir "$CACHE_DIR"
        --num_frames "$num_frames"
        --tcot_type "$tcot"
        --prompt_variant "$PROMPT_VARIANT"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --seed "$SEED"
        "${extra[@]}"
    )
    if [[ "$interleaved" == "1" ]]; then
        cmd+=(--interleaved_timestamps)
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  model=$model  frames=$num_frames  tcot=$tcot  interleaved=$interleaved  prompt=$PROMPT_VARIANT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "${cmd[@]}"
}

run_quantized() {
    # Same as run() but adds --quantize flag for 7B/8B models
    run "$1" "$2" "$3" "$4" "${@:5}" --quantize
}

run_bf16_large() {
    # Run large models in BF16 (no quantize) with expandable segments to reduce OOM risk
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run "$1" "$2" "$3" "$4" "${@:5}"
}


table2_zeroshot_eval() {
    # Run zero-shot eval for a given model, frame count, and CoT type
    local model="Qwen/Qwen3-VL-2B-Instruct"
    for SEED in "${SEEDS[@]}"; do
        run "$model" "$BEST_N" "$TCOT" "1"
    done

    local model="Qwen/Qwen3-VL-8B-Instruct"
    for SEED in "${SEEDS[@]}"; do
        run_quantized "$model" "$BEST_N" "$TCOT" "1"
    done

    local model="Qwen/Qwen2.5-VL-7B-Instruct"
    for SEED in "${SEEDS[@]}"; do
        run_quantized "$model" "$BEST_N" "$TCOT" "1"
    done

    local model="OpenGVLab/InternVL3-2B"
    for SEED in "${SEEDS[@]}"; do
        run "$model" "$BEST_N" "$TCOT" "0"
    done

    local model="OpenGVLab/InternVL3-8B"
    for SEED in "${SEEDS[@]}"; do
        run_quantized "$model" "$BEST_N" "$TCOT" "0"
    done
}



# ── Table 2: Frame count & interleaving ablation ──────────────────────────────
table2_video_frame_ablation() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — Frame count & interleaving ablation (Qwen3-VL-2B)"
    echo "======================================================================="
    
    local cot="none"
    local PROMPT_VARIANT="p6" 
    SEEDS=(43 44)  # Reset seeds for this ablation
    FRAMES=(4 16 32)

    for SEED in "${SEEDS[@]}"; do
        local model="Qwen/Qwen3-VL-2B-Instruct"
        for frames in "${FRAMES[@]}"; do
            run "$model" "$frames" "$cot" "1"   # interleaved timestamps
        done

        for model in "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen3-VL-8B-Instruct"; do
            for frames in "${FRAMES[@]}"; do
                run_quantized "$model" "$frames" "$cot" "1"   # interleaved timestamps
            done
        done
    done
}

table2_video_frame_ablation2() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — Frame count & interleaving ablation (Qwen3-VL-2B)"
    echo "======================================================================="
    
    local cot="none"
    local PROMPT_VARIANT="p6" 
    SEEDS=(42)
    for SEED in "${SEEDS[@]}"; do
        # local model="Qwen/Qwen3-VL-2B-Instruct"
        # for frames in 16 32; do
        #     run "$model" "$frames" "$cot" "1"   # interleaved timestamps
        # done

        for model in "Qwen/Qwen3-VL-8B-Instruct"; do
            for frames in 4 16 32; do
                run_quantized "$model" "$frames" "$cot" "1"   # interleaved timestamps
            done
        done
    done
}

table2_interleaved_ablation() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — Frame count & interleaving ablation (Qwen3-VL-2B)"
    echo "======================================================================="
    
    local cot="none"
    local PROMPT_VARIANT="p6" 
    SEEDS=(42 43 44)
    for SEED in "${SEEDS[@]}"; do
        # local model="Qwen/Qwen3-VL-2B-Instruct" 
        # for frames in 4 16 32; do
        #     run "$model" "$frames" "$cot" "0"   # interleaved timestamps
        # done

        for model in "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen3-VL-8B-Instruct"; do
            for frames in 8; do
                run_quantized "$model" "$frames" "$cot" "0"   # interleaved timestamps
            done
        done
    done
}

table2_zeroshot_gazeoverlay() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — Gaze overlay ablation (Qwen2.5-VL-7B, ${BEST_N} frames)"
    echo "======================================================================="
    
    VIDEO_ROOT="data/videodata_256/clips_dots"
    local model="Qwen/Qwen2.5-VL-7B-Instruct"
    local frames="$BEST_N"
    local tcot="none"
    SEEDS=(43 44 42)

    for SEED in "${SEEDS[@]}"; do
        run_quantized "$model" "$frames" "$tcot" "1"
    done
}


table2_prompt_variant_ablation() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — Variant ablation"
    echo "======================================================================="
    local cot="none"
    local frames="8"
    local PROMPT_VARIANT="p1"
    run "Qwen/Qwen3-VL-2B-Instruct" "$frames" "$cot" "1"    
    run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Instruct" "$frames" "$cot" "1"  

    local PROMPT_VARIANT="p2"
    run "Qwen/Qwen3-VL-2B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Instruct" "$frames" "$cot" "1"

    local PROMPT_VARIANT="p3"
    run "Qwen/Qwen3-VL-2B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Instruct" "$frames" "$cot" "1"

    local PROMPT_VARIANT="p4"
    run "Qwen/Qwen3-VL-2B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Instruct" "$frames" "$cot" "1"

    local PROMPT_VARIANT="p5"
    run "Qwen/Qwen3-VL-2B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Instruct" "$frames" "$cot" "1"

    local PROMPT_VARIANT="p7"
    run "Qwen/Qwen3-VL-2B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Instruct" "$frames" "$cot" "1"
}



# ── Table 1: Model comparison ─────────────────────────────────────────────────
table1_model_comparison() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 1 — Model comparison (${BEST_N} frames, no CoT)"
    echo "======================================================================="
    local cot="none"
    local frames="$BEST_N"
    # Small models — BF16 (fits in 16 GB)
    # run "Qwen/Qwen3-VL-2B-Instruct"    "$frames" "$cot" "0"
    # run "Qwen/Qwen3-VL-2B-Instruct"    "$frames" "$cot" "1"
    # run "OpenGVLab/InternVL3-2B"       "$frames" "$cot" "0"
    # run "OpenGVLab/InternVL3-2B"       "$frames" "$cot" "1"  # interleaved timestamps

    # Larger models — 8-bit quantized
    # run_quantized "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "0"
    # run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "0"
    # run_quantized "OpenGVLab/InternVL3-8B"       "$frames" "$cot" "0"
    # interleaved versions of 7B/8B models quantized
    # run_quantized "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
    # run_quantized "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    # run_quantized "OpenGVLab/InternVL3-8B"       "$frames" "$cot" "1"
    run_quantized "Qwen/Qwen3-VL-8B-Thinking"       "$frames" "$cot" "1" 
    run_quantized "Qwen/Qwen3-VL-8B-Thinking"       "$frames" "$cot" "0" 

    # non-quantized BF16 versions (OOM without expandable segments, but runs with it — see run_bf16_large helper)
    # run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "0"
    # run_bf16_large "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "0"
    # run_bf16_large "OpenGVLab/InternVL3-8B"       "$frames" "$cot" "0"
    # run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
    # run_bf16_large "Qwen/Qwen2.5-VL-7B-Instruct" "$frames" "$cot" "1"
    # run_bf16_large "OpenGVLab/InternVL3-8B"       "$frames" "$cot" "1"
    run_bf16_large "Qwen/Qwen3-VL-8B-Thinking"   "$frames" "$cot" "1" --sample_n 100
    run_bf16_large "Qwen/Qwen3-VL-8B-Thinking"   "$frames" "$cot" "0" --sample_n 100

    # API models — stratified 10-sample cost check first, then 100-sample run
    # echo ""
    # echo "--- Gemini 2.0 Flash: 10-sample cost check ---"
    # run "gemini-2.0-flash" "$frames" "$cot" "0" --sample_n 10 --sample_seed "$SEED"

    # echo ""
    # echo "--- GPT-4o: 10-sample cost check ---"
    # run "gpt-4o" "$frames" "$cot" "1" --sample_n 3 --sample_seed "$SEED"
    # run "gpt-4o" "$frames" "$cot" "1" --sample_seed "$SEED"

    # Uncomment to scale API runs to 100 samples after verifying cost:
    # run "gemini-2.0-flash" "$frames" "$cot" "0" --sample_n 100 --sample_seed "$SEED"
    # run "gpt-4o"           "$frames" "$cot" "0" --sample_seed "$SEED"
}

table2_ablation_study_prompt() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — Model comparison (${BEST_N} frames, no CoT)"
    echo "======================================================================="
    local cot="none"
    local frames="8"

    # check prompt variants for the best model (Qwen3-VL-8B) to see if we can boost cross-recall without CoT
    PROMPT_VARIANT="p1"
    run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
    PROMPT_VARIANT="p2"
    run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
    PROMPT_VARIANT="p3"
    run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
    PROMPT_VARIANT="p4"
    run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
    PROMPT_VARIANT="p5"
    run_bf16_large "Qwen/Qwen3-VL-8B-Instruct"   "$frames" "$cot" "1"
}


table2_visualprompt_runs() {
    # python -m dataprep.generate_vcot_detections --ann_file features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json   --video_root data/videodata_256/clips   --num_frames 8   --video_duration 2.0   --vcot_labels "automated vehicle,white circle"   --vcot_detector groundingdinotiny   --render_overlay_videos   --render_visuals som_overlay_nobg --overwrite --overwrite_rendered

    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — SOM overlay runs"
    echo "======================================================================="

    local cot="none"
    local model="Qwen/Qwen2.5-VL-7B-Instruct"
    local frames="$BEST_N"
    SEEDS=(42 43 44)
    for SEED in "${SEEDS[@]}"; do
        run_quantized "$model" "$frames" "$cot" "1" --vcot_visual som_overlay_nobg
    done

    # # local model="Qwen/Qwen3-VL-8B-Instruct"
    # run "$model" "$frames" "$cot" "1" --vcot_visual som_overlay_nobg
    # run "$model" "$frames" "$cot" "0" --vcot_visual som_overlay_nobg


    # local cot="tcot1"

    # local model="Qwen/Qwen2.5-VL-7B-Instruct"
    # run_quantized "$model" "$frames" "$cot" "1" --vcot_visual som_overlay_nobg
    # run_quantized "$model" "$frames" "$cot" "0" --vcot_visual som_overlay_nobg

    # local model="Qwen/Qwen3-VL-8B-Instruct"
    # run "$model" "$frames" "$cot" "1" --vcot_visual som_overlay_nobg
    # run "$model" "$frames" "$cot" "0" --vcot_visual som_overlay_nobg
}


# ── CoT / prompt ablation ───────────────────────────────────────────
table2_cot_ablation() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 2 — CoT ablation (Qwen2.5-VL-7B, ${BEST_N} frames)"
    echo "======================================================================="
    
    local model="Qwen/Qwen2.5-VL-7B-Instruct"
    local frames="$BEST_N"
    SEEDS=(42 43 44)
    for SEED in "${SEEDS[@]}"; do
        for cot in tcot1; do
            run_quantized "$model" "$frames" "$cot" "1"
        done
    done

    SEEDS=(43 44)
    for SEED in "${SEEDS[@]}"; do
        for cot in tcot4; do
            run_quantized "$model" "$frames" "$cot" "1"
        done
    done
}


table4_visual_cot_ablation() {
    echo ""
    echo "======================================================================="
    echo "  TABLE 4 — Visual CoT ablation (Qwen2.5-VL-7B, ${BEST_N} frames)"
    echo "======================================================================="
    local model="Qwen/Qwen2.5-VL-7B-Instruct"
    local frames="$BEST_N"
    local tcot="none"

    run_quantized "$model" "$frames" "$tcot" "1" --vcot_visual gaze_dot
    run_quantized "$model" "$frames" "$tcot" "1" --vcot_visual bbox_overlay
    run_quantized "$model" "$frames" "$tcot" "1" --vcot_visual som_overlay_bg
    run_quantized "$model" "$frames" "$tcot" "1" --vcot_text bbox_coords
    run_quantized "$model" "$frames" "$tcot" "1" --vcot_visual bbox_overlay --vcot_text bbox_coords
    run_quantized "$model" "$frames" "$tcot" "1" --vcot_visual som_overlay_bg --vcot_text bbox_coords
}

table5_context_guided() {
    bash scripts/run_contextual_input_evals.sh
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "$STEP" in
    frame_ablation)
        table2_video_frame_ablation
        ;;
    model_comparison)
        table1_model_comparison
        ;;
    cot_ablation)
        table2_cot_ablation
        ;;
    visual_cot_ablation)
        table4_visual_cot_ablation
        ;;
    contextual_inputs)
        table5_context_guided
        ;;
    ablation_study)
        table2_video_frame_ablation
        table2_prompt_variant_ablation
        ;;
    test)
        table2_visualprompt_runs
        table5_context_guided
        ;;
    table2)
        table2_zeroshot_eval
        ;;
    table2_videoframe)
        table2_video_frame_ablation2
        ;; 
    table2_interleaved)
        table2_interleaved_ablation
        ;; 
    cot_visual_runs)
        table2_visualprompt_runs
        table2_cot_ablation
        ;;
    all)
        table2_video_frame_ablation
        echo ""
        echo ">>> Frame ablation done. Update BEST_N based on Table 2 results"
        echo ">>> then run: BEST_N=<N> bash scripts/run_vlm_eval_grid.sh model_comparison"
        ;;
    *)
        echo "Unknown step: $STEP. Choose from: frame_ablation model_comparison cot_ablation visual_cot_ablation contextual_inputs all"
        exit 1
        ;;
esac

echo ""
echo "Done."


TOPIC="aicheetah_ntfy_"

curl -d "✅ $STEP is done!" \
    -H "Zero-shot evaluation" \
    -H "Priority: high" \
    -H "Tags: white_check_mark,computer" \
    ntfy.sh/$TOPIC