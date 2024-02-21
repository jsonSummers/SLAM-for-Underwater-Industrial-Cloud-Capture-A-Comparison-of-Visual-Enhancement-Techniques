#!/bin/bash

file_names=("$@")

eps="0.1"
min_points="10"
gen_silhouette="False"

python_script="./cluster.py"

for name in "${file_names[@]}"; do
    for test in {1..3}; do
        input_csv_file="../Results/${name}/${test}/MapPoints_${name}.csv"
        input_ply_file="../Results/${name}/${test}/${name}.ply"
        output_csv_file="../Results/${name}/${test}/clusters_${name}.csv"
        echo "Processing $name..."
        python3 "$python_script" "$input_csv_file" "$input_ply_file" "$eps" "$min_points" "$gen_silhouette" "$output_csv_file"
    done
done

echo "Processing complete."