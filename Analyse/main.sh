#!/bin/bash

file_names=("original" "ancuti2017color" "demir2023low" "islam2020fast" "li2019underwater" "sharma2023wavelength")

echo "Converting ORB SLAM CSV files to PLY"
#./convert.sh "${file_names[@]}"

echo "Calculating cluster information"
#./cluster.sh "${file_names[@]}"

echo "Performing full analysis"
./analyse.sh "${file_names[@]}"

echo "Processing complete."