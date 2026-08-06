
# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# sample_fpss=(5 10 15)
# temperatures=(0)
# anno_files=(features/groundvqa/annotations.VRbinary_00000_test_close.json 
#             features/groundvqa/annotations.VRbinary_00100_test_close.json
#             features/groundvqa/annotations.VRbinary_00010_test_close.json
#             features/groundvqa/annotations.VRbinary_00001_test_close.json)

# for sample_fps in "${sample_fpss[@]}"; do
#     for model_name in "${model_name[@]}"; do
#         for temperature in "${temperatures[@]}"; do
#             for anno_file in "${anno_files[@]}"; do
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file
#             done
#         done
#     done
# done    



# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# sample_fpss=(10)
# temperatures=(0)
# everys=(3 5 6 15)
# setups=(00100)
# anno_files=(features/groundvqa/annotations.VRbinary_00100_every{}_test_close.json
#             features/groundvqa/annotations.VRbinary_00010_test_close.json)

# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#          for every in "${everys[@]}"; do
#              for model_name in "${model_name[@]}"; do
#                  for temperature in "${temperatures[@]}"; do
#                     anno_file=features/groundvqa/annotations.VRbinary_${setup}_every${every}_test_close.json
#                         python -m training.eval_qwen \
#                             --model_name $model_name \
#                             --temperature $temperature \
#                             --sample_fps $sample_fps \
#                             --ann_file $anno_file
#                  done
#              done
#          done
#     done
# done



# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# sample_fpss=(10)
# temperatures=(0)
# everys=(1 2 3 5 6 15)
# setups=(00010)

# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#         for every in "${everys[@]}"; do
#             for model_name in "${model_name[@]}"; do
#                 for temperature in "${temperatures[@]}"; do
#                 anno_file=features/groundvqa/annotations.VRbinary_${setup}_every${every}_test_close.json
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file
#                 done
#             done
#         done
#     done
# done


# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# sample_fpss=(10)
# temperatures=(0)

# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#         for every in "${everys[@]}"; do
#             for model_name in "${model_name[@]}"; do
#                 for temperature in "${temperatures[@]}"; do
#                 anno_file=features/groundvqa/annotations.VRbinary_00000_test_close.json
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file
#                 done
#             done
#         done
#     done
# done


#### ✅ generalizationwith CoT prompting (==>cot)
# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# setups=(00000 00100 00010 00001)
# sample_fpss=(2)
# temperatures=(0)

# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#         for model_name in "${model_name[@]}"; do
#             for temperature in "${temperatures[@]}"; do
#                 anno_file=features/groundvqa/annotations.VRbinary_${setup}_test_close.json
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file
#             done
#         done
#     done
# done


#### ✅ generalizationwith CoT prompting, with more detailed CoT instruction (==>cot2)
# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# setups=(00000 00100 00010 00001)
# sample_fpss=(2)
# temperatures=(0)

# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#         for model_name in "${model_name[@]}"; do
#             for temperature in "${temperatures[@]}"; do
#                 anno_file=features/groundvqa/annotations.VRbinary_${setup}_test_close.json
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file
#             done
#         done
#     done
# done


# model_name=Qwen/Qwen3-VL-2B-Instruct
# temperature=0
# sample_fps=2
# setup=00001
# anno_file=features/groundvqa_prompt/annotations.VRbinary_${setup}_test_close.json
# python -m training.eval_qwen \
#     --model_name $model_name \
#     --temperature $temperature \
#     --sample_fps $sample_fps \
#     --ann_file $anno_file



# model_name=(Qwen/Qwen3-VL-2B-Instruct)
# setups=(00000)
# sample_fpss=(2)
# temperatures=(0)

# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#         for model_name in "${model_name[@]}"; do
#             for temperature in "${temperatures[@]}"; do
#                 anno_file=features/groundvqa/annotations.VRbinary_${setup}_test_close.json
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file \
#                     --video_root data/videodata_256/clips_overlay

#                 anno_file=features/groundvqa_overlay/annotations.VRbinary_${setup}_test_close.json
#                 python -m training.eval_qwen \
#                     --model_name $model_name \
#                     --temperature $temperature \
#                     --sample_fps $sample_fps \
#                     --ann_file $anno_file \
#                     --video_root data/videodata_256/clips_overlay
#             done
#         done
#     done
# done



model_name=(Qwen/Qwen3-VL-2B-Instruct)
setups=(00000)
# setups=(00000 00010_every10 00100_every10 00001)
# setups=(00010_every10)
sample_fpss=(2)
temperatures=(0)
# logdirs=(logs/qwen_new_qn3_cot1_eval logs/qwen_new_qn3_cot4_eval logs/qwen_new_qn3_cot5_eval logs/qwen_new_qn3_cot6_eval)
logdirs=(logs/qwen_new_qn3_cot1_eval)
TOPIC="aicheetah_ntfy_"


for sample_fps in "${sample_fpss[@]}"; do
    for setup in "${setups[@]}"; do
        for model_name in "${model_name[@]}"; do
            for temperature in "${temperatures[@]}"; do
                for logdir in "${logdirs[@]}"; do
                    anno_file=features/groundvqa_qn5/annotations.VRbinary_${setup}_test_close.json
                    python -m training.eval_qwen \
                        --model_name $model_name \
                        --temperature $temperature \
                        --sample_fps $sample_fps \
                        --ann_file $anno_file \
                        --logdir $logdir \
                        --video_root data/videodata_256/clips_bone

                    curl -d "✅ Script (bone overlay) on $(hostname) is finished!" \
                        -H "Title: Task Complete" \
                        -H "Priority: high" \
                        -H "Tags: white_check_mark,computer" \
                        ntfy.sh/$TOPIC

                    python -m training.eval_qwen \
                        --model_name $model_name \
                        --temperature $temperature \
                        --sample_fps $sample_fps \
                        --ann_file $anno_file \
                        --logdir $logdir \
                        --video_root data/videodata_256/clips_rainbow

                    curl -d "✅ Script (rainbow overlay) on $(hostname) is finished!" \
                        -H "Title: Task Complete" \
                        -H "Priority: high" \
                        -H "Tags: white_check_mark,computer" \
                        ntfy.sh/$TOPIC
                done
            done
        done
    done
done

# setups=(00000)
# for sample_fps in "${sample_fpss[@]}"; do
#     for setup in "${setups[@]}"; do
#         for model_name in "${model_name[@]}"; do
#             for temperature in "${temperatures[@]}"; do
#                 for logdir in "${logdirs[@]}"; do
#                     anno_file=features/groundvqa_qn5/annotations.VRbinary_${setup}_test_close.json
#                     python -m training.eval_qwen \
#                         --model_name $model_name \
#                         --temperature $temperature \
#                         --sample_fps $sample_fps \
#                         --ann_file $anno_file \
#                         --logdir $logdir \
#                         --video_root data/videodata_256/clips_overlay
#                 done
#             done
#         done
#     done
# done




# model_name=Qwen/Qwen3-VL-2B-Instruct
# temperature=0
# sample_fps=2
# setup=00001
# anno_file=features/groundvqa_prompt/annotations.VRbinary_${setup}_test_close.json
# python -m training.eval_qwen \
#     --model_name $model_name \
#     --temperature $temperature \
#     --sample_fps $sample_fps \
#     --ann_file $anno_file


# model_name=Qwen/Qwen3-VL-2B-Instruct
# temperature=0
# sample_fps=2
# setup=(00000 00001)
# for s in "${setup[@]}"; do
#     anno_file=features/groundvqa_qn2/annotations.VRbinary_${s}_test_close.json
#     python -m training.eval_qwen \
#         --model_name $model_name \
#         --temperature $temperature \
#         --sample_fps $sample_fps \
#         --ann_file $anno_file
# done