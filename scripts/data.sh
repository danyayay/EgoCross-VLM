tasks=(crossing_intention)

for task in "${tasks[@]}"; do
    python -m dataprep.d2split_annos --anno_filename "annotations.VRbinary__${task}__full_close.json"
done
