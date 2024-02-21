#!/bin/bash

file_names=("$@")

python_script="./convert_to_ply.py"

for name in "${file_names[@]}"; do
    for test in {1..3}; do
        input_path="../Results/${name}/${test}/MapPoints_${name}.csv"
        output_path="../Results/${name}/${test}/${name}.ply"
        echo "Processing $name..."
        python3 "$python_script" "$input_path" "$output_path"
    done
done

echo "Processing complete."