# list of files in the given directory
model_name="Qwen3-VL-2B-Instruct"
files=$(ls logs/qwen_train/$model_name/)

for file in $files; do
    python scripts/parse_qwentrain_log.py --log_file "logs/qwen_train/$model_name/$file/train.log"
done