models=('transformer' 'cross_attention' 'lstm')
metrics=('acc' 'macro_f1' 'loss')

visual_encoder="clip"
for model in "${models[@]}"; do
    for metric in "${metrics[@]}"; do
        python -m training.train_fusion --visual_encoder $visual_encoder --model $model --eval_metric $metric
    done
done

visual_encoder='cnn'
for model in "${models[@]}"; do
    for metric in "${metrics[@]}"; do
        python -m training.train_fusion --visual_encoder $visual_encoder --model $model --eval_metric $metric
    done
done


python -m training.train_dl --visual_encoder clip --model hybrid --eval_metric acc



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




#### Training DL
# problem = multi / binary 
# visual_encoder = pixels / simple / clip 
# features = use / not use additional numeric features

# python -m training.train_fusion --annotations 'features/features/annotations.VRmulti_test_close.json' --visual-encoder pixels --dataset-type vr --hp-trials 150

# python -m training.train_fusion --annotations 'features/features/annotations.VRmulti_test_close.json' --n-classes 5 --visual-encoder simple --dataset-type vr --hp-trials 100 --hp-dashboard-port 7070 

# python -m training.train_fusion --annotations 'features/features/annotations.VRmulti_test_close.json' --visual-encoder clip --dataset-type vr --hp-trials 150 --hp-dashboard-port 6060



# python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder pixels --dataset-type vr --hp-trials 150 --hp-dashboard-port 1010

# python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder simple --dataset-type vr --hp-trials 150 --hp-dashboard-port 2020

# python -m training.train_fusion --annotations '../data/features/annotations.VRbin_test_close.json' --visual-encoder clip --dataset-type vr --hp-trials 150 --hp-dashboard-port 3030




# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder clip --no-extra-modalities --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_egocentric"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder simple --no-extra-modalities --hp-tune --hp-trials 100 --optuna_url "sqlite:///db.sqlite_egocentric"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder pixels --no-extra-modalities --hp-tune --hp-trials 100 --optuna_url "sqlite:///db.sqlite_egocentric"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder pixels --hp-tune --hp-trials 100 --optuna_url "sqlite:///db.sqlite_egocentric"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder simple --hp-tune --hp-trials 100 --optuna_url "sqlite:///db.sqlite_egocentric"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder clip --crop-anchor saliency --no-extra-modalities --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_egocentric"


# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder cnn --model hybrid --crop-anchor padding --no-extra-modalities --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder cnn --model hybrid --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder clip --model hybrid --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder cnn --model hybrid --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder clip --model hybrid --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"


# # Hybrid, cnn backbone on videos, transformer-based model --> done
# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder cnn --model concat --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# # egocentric only, cnn on video feature extraction, lstm-based model --> done
# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder cnn --model hybrid --no-extra-modalities --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# # egocentric only, clip on video feature extraction, lstm-based model --> done
# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder clip --model hybrid --no-extra-modalities --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"

# # egocentric only, cnn on video feature extraction, transformer-based model --> next to run
# python -m training.train_fusion --annotations 'features/features/annotations.VRbin_test_close.json' --visual-encoder cnn --model concat --no-extra-modalities --crop-anchor padding --hp-tune --hp-trials 80 --optuna_url "sqlite:///db.sqlite_new"