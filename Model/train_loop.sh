#!/bin/bash

# Define experiments with corresponding number of negatives
declare -A experiments=( ["trip"]=1 ["quad"]=2 )
# ["quin"]=3 )

# Loop through each experiment
for experiment in "${!experiments[@]}"; do
    # Loop through each run (1, 2, 3) for the current experiment
    for run in {1..2}; do
        echo "Running experiment: $experiment, run: $run"
        # Define the save path
        save_path="experiments/$experiment/$run"
        num_negatives="${experiments[$experiment]}"
        # Create the directory if it doesn't exist
        mkdir -p "$save_path/checkpoints/enhancer"

        # Run the Python script with num_negatives and save_path as arguments
        python train.py "$num_negatives" "$save_path"
    done
done