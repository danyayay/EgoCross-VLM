# sleep 2.3h

# model_name=Qwen/Qwen3-VL-2B-Instruct
# sample_fpss=(2)
# datasetups=(00000)
# ft_types=('lora_llm_attn_mlp' 'lora_llm_vlm_bridger' 'lora_vlm_bridger')
# # lora_ranks=(2 4)
# # lora_alphas=(8 16 32)

# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         ann_file_template="features/groundvqa/annotations.VRbinary_${datasetup}_mode_close.json"
#         for ft_type in "${ft_types[@]}"; do
#             python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template
#         done
#     done
# done    

# model_name=Qwen/Qwen3-VL-2B-Instruct
# sample_fpss=(2)
# datasetups=(00100)
# ft_types=('lora_vlm_bridger')

# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         ann_file_template="features/groundvqa/annotations.VRbinary_${datasetup}_mode_close.json"
#         for ft_type in "${ft_types[@]}"; do
#             python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template
#         done
#     done
# done   



#### ✅ Train with gaze-overlayed videos on different ft types
# model_name=Qwen/Qwen3-VL-2B-Instruct
# sample_fpss=(2)
# datasetups=(00000)
# ft_types=('lora_llm_attn_qv' 'lora_llm_attn_qkvo' 'lora_llm_mlp' 'lora_llm_attn_mlp' 'lora_llm_vlm_bridger' 'lora_vlm_bridger')
# # ft_types=('lora_llm_attn_qv' 'lora_llm_mlp' 'lora_llm_attn_mlp')

# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         ann_file_template="features/groundvqa/annotations.VRbinary_${datasetup}_mode_close.json"
#         for ft_type in "${ft_types[@]}"; do
#             python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template \
#                 --video_root data/videodata_256/clips_overlay
#         done
#     done
# done

# ### Train with gaze-overlayed videos on different ft types (with more explanations)
# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         ann_file_template="features/groundvqa_overlay/annotations.VRbinary_${datasetup}_mode_close.json"
#         for ft_type in "${ft_types[@]}"; do
#             python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template \
#                 --video_root data/videodata_256/clips_overlay
#         done
#     done
# done



#### ✅ Train with gaze-overlayed videos on different ft types
# model_name=Qwen/Qwen3-VL-2B-Instruct
# sample_fpss=(2)
# datasetups=(00100 00010)
# everys=(1 2 3 5 6 10 15)
# # ft_types=('lora_llm_attn_qv' 'lora_llm_attn_qkvo' 'lora_llm_mlp' 'lora_llm_attn_mlp' 'lora_llm_vlm_bridger' 'lora_vlm_bridger')
# ft_types=('lora_llm_vlm_bridger')

# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         for every in "${everys[@]}"; do
#             ann_file_template="features/groundvqa/annotations.VRbinary_${datasetup}_every${every}_mode_close.json"
#             for ft_type in "${ft_types[@]}"; do
#                 python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template \
#                 --video_root data/videodata_256/clips
#             done
#         done
#     done
# done

### something missed out
# model_name=Qwen/Qwen3-VL-2B-Instruct
# sample_fpss=(2)
# datasetups=(00100)
# everys=(3 15)
# # ft_types=('lora_llm_attn_qv' 'lora_llm_attn_qkvo' 'lora_llm_mlp' 'lora_llm_attn_mlp' 'lora_llm_vlm_bridger' 'lora_vlm_bridger')
# ft_types=('lora_llm_vlm_bridger')

# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         for every in "${everys[@]}"; do
#             ann_file_template="features/groundvqa/annotations.VRbinary_${datasetup}_every${every}_mode_close.json"
#             for ft_type in "${ft_types[@]}"; do
#                 python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template \
#                 --video_root data/videodata_256/clips
#             done
#         done
#     done
# done

# model_name=Qwen/Qwen3-VL-2B-Instruct
# sample_fpss=(2)
# datasetups=(00010)
# everys=(5 6)
# # ft_types=('lora_llm_attn_qv' 'lora_llm_attn_qkvo' 'lora_llm_mlp' 'lora_llm_attn_mlp' 'lora_llm_vlm_bridger' 'lora_vlm_bridger')
# ft_types=('lora_llm_vlm_bridger')

# for sample_fps in "${sample_fpss[@]}"; do
#     for datasetup in "${datasetups[@]}"; do
#         for every in "${everys[@]}"; do
#             ann_file_template="features/groundvqa/annotations.VRbinary_${datasetup}_every${every}_mode_close.json"
#             for ft_type in "${ft_types[@]}"; do
#                 python -m training.train_qwen \
#                 --model_name $model_name \
#                 --ft_type $ft_type \
#                 --sample_fps $sample_fps \
#                 --ann_file_template $ann_file_template \
#                 --video_root data/videodata_256/clips
#             done
#         done
#     done
# done



### eval trained model
python -m training.train_qwen \
    --ann_file_template features/groundvqa/annotations.VRbinary_00100_every15_mode_close.json \
    --adapter_ckpt_path logs/qwen_train/Qwen3-VL-2B-Instruct/20260316_094942 \
    --ft_type lora_llm_vlm_bridger \
    --sample_fps 2 \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --mode eval 

# for sample_fpss in "${sample_fpss[@]}"; do
#     for eval_deterministic in "${eval_deterministics[@]}"; do
#         if [ "$eval_deterministic" = true ]; then
#             eval_deterministic_flag="--eval_deterministic"
#         else
#             eval_deterministic_flag=""
#         fi
#         for ft_type in "${ft_types[@]}"; do
#             for lora_rank in "${lora_ranks[@]}"; do
#                 for lora_alpha in "${lora_alphas[@]}"; do
#                     python -m training.train_qwen \
#                         --model_name $model_name \
#                         --ft_type $ft_type \
#                         --sample_fps $sample_fpss \
#                         $eval_deterministic_flag \
#                         --lora_rank $lora_rank \
#                         --lora_alpha $lora_alpha
#                 done
#             done
#         done
#     done
# done    


# lora_ranks=(8)
# lora_alphas=(8 32)

# for sample_fpss in "${sample_fpss[@]}"; do
#     for eval_deterministic in "${eval_deterministics[@]}"; do
#         if [ "$eval_deterministic" = true ]; then
#             eval_deterministic_flag="--eval_deterministic"
#         else
#             eval_deterministic_flag=""
#         fi
#         for ft_type in "${ft_types[@]}"; do
#             for lora_rank in "${lora_ranks[@]}"; do
#                 for lora_alpha in "${lora_alphas[@]}"; do
#                     python -m training.train_qwen \
#                         --model_name $model_name \
#                         --ft_type $ft_type \
#                         --sample_fps $sample_fpss \
#                         $eval_deterministic_flag \
#                         --lora_rank $lora_rank \
#                         --lora_alpha $lora_alpha
#                 done
#             done
#         done
#     done
# done  

# python -m training.train_qwen \
#     --model_name $model_name \
#     --ft_type 'lora_llm_vlm_bridger' \
#     --sample_fps 2.0 \
#     --eval_deterministic \
#     --lora_rank 2 \
#     --lora_alpha 8
