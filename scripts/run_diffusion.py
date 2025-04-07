#!/usr/bin/env python3
"""Script to run the diffusion model training."""

import argparse
import os
import sys

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Run diffusion model training")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/federico/results",
        help="Directory to save results and visualizations",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000, help="Number of data samples to generate"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    return parser.parse_args()


def main():
    """Run the diffusion model training."""
    # Parse arguments
    args = parse_args()
    results_dir = args.results_dir
    n_samples = args.n_samples
    epochs = args.epochs

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    try:
        # Import the training module directly
        from training.train_diffusion import main as train_main

        # Override sys.argv to pass arguments to the training script
        sys.argv = [
            sys.argv[0],
            "--n_samples",
            str(n_samples),
            "--sigma",
            "0.1",
            "--time_steps",
            "100",  # More diffusion steps
            "--epochs",
            str(epochs),
            "--batch_size",
            "128",
            "--learning_rate",
            "1e-3",
            "--save_dir",
            results_dir,
            "--seed",
            "42",
        ]

        # Run the training
        print(
            f"Running diffusion model training with {n_samples} samples and {epochs} epochs..."
        )
        print(f"Results will be saved to: {results_dir}")
        train_main()

        # Display results
        print(f"Training completed! Results saved to {results_dir}")
        print("Generated images:")
        images = [f for f in os.listdir(results_dir) if f.endswith(".png")]

        if images:
            for img in images:
                print(f"{results_dir}/{img}")
        else:
            print("No images found. Training may have failed.")

    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Make sure you're running this script from the project root directory.")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
