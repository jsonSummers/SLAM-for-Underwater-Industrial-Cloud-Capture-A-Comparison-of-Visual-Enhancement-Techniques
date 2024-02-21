#!/bin/bash

experiment_names=("$@")

python_script="./analyse.py"
file_paths=()
output_csv_file="../Results/final_result.csv"


for experiment in "${experiment_names[@]}"; do
    for run in {1..3}; do
        input_csv_file="../Results/${experiment}/${run}/clusters_${experiment}.csv"
        file_paths+=("$input_csv_file")
    done
done

echo "Processing files..."

python3 "$python_script" "${file_paths[@]}" "$output_csv_file"

echo "Processing complete."