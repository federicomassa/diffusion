#!/bin/bash

# Define results directory and training parameters
RESULTS_DIR="/home/federico/results"
N_SAMPLES=5000
EPOCHS=30
TIME_STEPS=100

# Create results directory
mkdir -p "$RESULTS_DIR"

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Run the diffusion training script directly using Python
echo "Running diffusion model training with $N_SAMPLES samples and $EPOCHS epochs..."
echo "Results will be saved to: $RESULTS_DIR"

# If running from Bazel, use the Python interpreter from the runfiles
if [[ -n "$RUNFILES_DIR" ]]; then
    # We're running from Bazel
    echo "Running from Bazel runfiles directory: $RUNFILES_DIR"
    cd "$PROJECT_ROOT"
    python3 -m training.train_diffusion \
        --n_samples $N_SAMPLES \
        --sigma 0.1 \
        --time_steps $TIME_STEPS \
        --epochs $EPOCHS \
        --batch_size 128 \
        --learning_rate 1e-3 \
        --save_dir "$RESULTS_DIR" \
        --seed 42
else
    # We're running directly from the command line
    cd "$PROJECT_ROOT"
    python3 -m training.train_diffusion \
        --n_samples $N_SAMPLES \
        --sigma 0.1 \
        --time_steps $TIME_STEPS \
        --epochs $EPOCHS \
        --batch_size 128 \
        --learning_rate 1e-3 \
        --save_dir "$RESULTS_DIR" \
        --seed 42
fi

# Display results
echo "Training completed! Results saved to $RESULTS_DIR"
echo "Generated images:"
ls -la "$RESULTS_DIR"/*.png 2>/dev/null || echo "No images found. Training may have failed." 