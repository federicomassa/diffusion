# Diffusion

A Python project for diffusion models.

## Building with Bazel

This project uses Bazel as the build system. 

### Prerequisites

- Install [Bazel](https://bazel.build/install)
- Python 3.12

### Building and Running

Build all targets:
```bash
bazel build //...
```

Run a specific binary:
```bash
bazel run //:check_gpu
bazel run //:test_keras_install
```

### Development

To add a new dependency, add it to `third_party/requirements.txt` and run:
```bash
bazel sync --only=pip_deps
```

## Diffusion Model Implementation

This project implements a basic diffusion model in Keras, trained on a multimodal Gaussian distribution in 2D.

### Project Structure

- `data_utils/`: Data generation utilities
  - `gaussian_data.py`: Generates multimodal Gaussian data in 2D
- `models/`: Model implementations
  - `diffusion.py`: Basic implementation of a DDPM (Denoising Diffusion Probabilistic Model)
- `training/`: Training scripts
  - `train_diffusion.py`: Script to train the diffusion model
- `scripts/`: Helper scripts
  - `run_diffusion.py`: Python script to run the diffusion model training

### Training the Model with Bazel

You can run the diffusion model training using Bazel with:

```bash
bazel run //scripts:run_diffusion -- --results-dir=/path/to/results --n-samples=5000 --epochs=30
```

Or run the training module directly:

```bash
bazel run //training:train_diffusion -- \
    --n_samples 5000 \
    --sigma 0.1 \
    --time_steps 100 \
    --epochs 30 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --save_dir /path/to/results \
    --seed 42
```

By default, results will be stored in `/home/federico/results`. Both methods will:
1. Generate data from a multimodal Gaussian distribution (with 4 modes)
2. Train a diffusion model on this data (using 30 epochs and 5000 samples by default)
3. Generate samples from the trained model
4. Save visualizations to the specified results directory

### Available Bazel Targets

The project includes the following Bazel targets:

- **Data utilities**:
  - `//data_utils:data_utils` - Base data utilities
  - `//data_utils:gaussian_data` - Multimodal Gaussian data generator

- **Models**:
  - `//models:models` - Base models
  - `//models:diffusion` - Diffusion model implementation

- **Training**:
  - `//training:training` - Base training utilities
  - `//training:train_diffusion` - Training module for diffusion models

- **Scripts**:
  - `//scripts:run_diffusion` - Python script to run the diffusion model training