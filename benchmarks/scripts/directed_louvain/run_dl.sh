#!/bin/bash

# Define the path.
user_path="/Users/alissachavalithumrong/Documents/research/flowcommunities"

# Loop over a set of mu values
for mu_name in 10 20 30 40 50 60 70 80 90; do

    for i in $(seq 1 100); do
        # Define the file paths
        input_file="${user_path}/benchmarks/LF_created_networks/500_node/${mu_name}/${i}_network_N500_k50_maxk75_mu${mu_name}.dat"
        output_file="${user_path}/benchmarks/directed_louvain/500_node/${mu_name}/${i}_mu${mu_name}.tree"

        # Run the community detection command
        ./bin/community -f "${input_file}" -l -1 -v > "${output_file}"

        # Modify the output tree file
        sed -i '' '1d' "${output_file}"  # Delete the first line
        awk '/-1 -1/ {exit} {print}' "${output_file}" > temp.tree && mv temp.tree "${output_file}"  # Delete everything after the first occurrence of "-1 -1"
    done

done