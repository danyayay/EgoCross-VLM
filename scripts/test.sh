TOPIC="aicheetah_ntfy_"

bash scripts/run_vlm_training_effect_evals.sh
bash scripts/run_vlm_training_effect_evals2.sh
bash scripts/run_contextual_input_evals.sh
# python -m training.eval_vlm \
#   --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#   --quantize \
#   --ann_file "features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json" \
#   --video_root "data/videodata_256/clips" \
#   --num_frames "8" \
#   --max_new_tokens "1024" \
#   --prompt_variant "p6" \
#   --tcot_type "tcot4" \
#   --vcot_visual "som_overlay" \
#   --interleaved_timestamps \
#   --log_dir "logs/vlm_eval_newformat_100" \
#   --sample_n "100"


# python -m training.eval_vlm \
#   --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#   --quantize \
#   --ann_file "features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json" \
#   --video_root "data/videodata_256/clips" \
#   --num_frames "4" \
#   --max_new_tokens "1024" \
#   --prompt_variant "p6" \
#   --tcot_type "none" \
#   --vcot_visual "som_overlay_nobg" \
#   --interleaved_timestamps \
#   --log_dir "logs/vlm_eval_newformat" \

# python -m training.eval_vlm \
#   --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#   --quantize \
#   --ann_file "features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json" \
#   --video_root "data/videodata_256/clips" \
#   --num_frames "4" \
#   --max_new_tokens "1024" \
#   --prompt_variant "p6" \
#   --tcot_type "tcot10" \
#   --vcot_visual "som_overlay_nobg" \
#   --interleaved_timestamps \
#   --log_dir "logs/vlm_eval_newformat" \


# python -m training.eval_vlm \
#   --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#   --quantize \
#   --ann_file "features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json" \
#   --video_root "data/videodata_256/clips" \
#   --num_frames "8" \
#   --max_new_tokens "1024" \
#   --prompt_variant "p1" \
#   --tcot_type "none" \
#   --vcot_visual "som_overlay_nobg" \
#   --interleaved_timestamps \
#   --log_dir "logs/vlm_eval_newformat" \

# python -m training.eval_vlm \
#   --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#   --quantize \
#   --ann_file "features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json" \
#   --video_root "data/videodata_256/clips" \
#   --num_frames "8" \
#   --max_new_tokens "1024" \
#   --prompt_variant "p6" \
#   --tcot_type "tcot1" \
#   --vcot_visual "som_overlay_nobg" \
#   --interleaved_timestamps \
#   --log_dir "logs/vlm_eval_newformat" \

# python -m training.eval_vlm \
#   --model_name "Qwen/Qwen2.5-VL-7B-Instruct" \
#   --quantize \
#   --ann_file "features/groundvqa_qn3/annotations.VRbinary_00000_test_close.json" \
#   --video_root "data/videodata_256/clips" \
#   --num_frames "8" \
#   --max_new_tokens "1024" \
#   --prompt_variant "p6" \
#   --tcot_type "tcot4" \
#   --vcot_visual "som_overlay_nobg" \
#   --interleaved_timestamps \
#   --log_dir "logs/vlm_eval_newformat" \



curl -d "✅ Done!" \
    -H "Title: Task Complete" \
    -H "Priority: high" \
    -H "Tags: white_check_mark,computer" \
    ntfy.sh/$TOPIC