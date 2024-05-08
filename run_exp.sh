#!/bin/bash

# Run synthetic data
# List of possible arguments
arguments=(
    "--config=default.yaml"
    "--config=high_dim.yaml"
    "--config=low_cor.yaml"
)

# Loop through the arguments and run synthetic.py
for arg in "${arguments[@]}"; do
    python synthetic.py "$arg"
done
