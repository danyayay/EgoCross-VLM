# list of files in the given directory
model_name="Qwen3-VL-2B-Instruct"
files=$(ls logs/qwen_eval/$model_name/)

for file in $files; do
    if [[ $file == *.log ]]; then
        python scripts/parse_qweneval_log.py --log-file "logs/qwen_eval/$model_name/$file"
    fi
done