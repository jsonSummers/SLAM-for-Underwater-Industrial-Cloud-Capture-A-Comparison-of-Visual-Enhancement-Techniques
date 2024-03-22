#!/bin/bash

declare -A experiments=( ["trip"]=1 ["quad"]=2 ["quin"]=3 )

for experiment in "${!experiments[@]}"; do
    for run in {1..3}; do
        save_path="experiments/$experiment/$run"
        num_negatives="${experiments[$experiment]}"
        mkdir -p "$save_path"

        python train.py "$num_negatives" "$save_path"
    done
done