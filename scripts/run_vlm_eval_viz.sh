# To run:
# bash scripts/run_vlm_eval_viz.sh


SEEDS=(42 43 44)

# for raw video input
# python -m utils.summarize_context_guided --log_root logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct --out summary.csv
# for SEED in "${SEEDS[@]}"; do
#     python -m utils.visualize_context_guided --summary logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct/seed_${SEED}/summary/summary.csv --out logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct/seed_${SEED}/summary
# done
# python -m utils.visualize_cont_sampling_rate_ablation --logdir logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct --band_mode std

python -m utils.visualize_zeroshot_videoframe-interleaved_ablation --logdir logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct --band_mode std
python -m utils.visualize_zeroshot_cntx_comparison --log_root logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct --out_dir logs/vlm_eval_context_guided/Qwen2.5-VL-7B-Instruct

# for gaze overlaid video input
# summarize_context_guided auto-discovers seed_*/ under --log_root, writes a
# per-seed summary.csv under seed_<N>/summary/, and writes a cross-seed
# mean+-std aggregate to <log_root>/summary/summary_aggregated.csv.
python -m utils.summarize_context_guided --log_root logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct --out summary.csv
for SEED in "${SEEDS[@]}"; do
    python -m utils.visualize_context_guided --summary logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct/seed_${SEED}/summary/summary.csv --out logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct/seed_${SEED}/summary
done
# visualize_context_guided also accepts summary_aggregated.csv directly (mean+-std
# columns are normalized to the plain metric names internally).
python -m utils.visualize_context_guided --summary logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct/summary/summary_aggregated.csv --out_dir logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct/summary
python -m utils.visualize_cont_sampling_rate_ablation --logdir logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct --band_mode std






python -m utils.summarize_zeroshot_cntx_comparison --log_root logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct --out summary.csv
python -m utils.visualize_zeroshot_cntx_comparison --summary logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct/summary/summary_aggregated.csv --out_dir logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct/summary
python -m utils.visualize_zeroshot_cont_sampling_rate_ablation --logdir logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct --band_mode std
python -m utils.visualize_zeroshot_cntx_comparison --log_root logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct --out_dir logs/vlm_eval_context_guided_dot/Qwen2.5-VL-7B-Instruct