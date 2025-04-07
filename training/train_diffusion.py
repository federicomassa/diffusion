"""Script to train a diffusion model on multimodal Gaussian data."""

import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_utils.gaussian_data import generate_multimodal_gaussian_data, visualize_data
from models.diffusion import DiffusionModel


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a diffusion model on multimodal Gaussian data"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of data samples to generate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Standard deviation for each Gaussian mode",
    )
    parser.add_argument(
        "--time_steps", type=int, default=200, help="Number of diffusion time steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Directory to save results and visualizations",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def setup_experiment(args: argparse.Namespace) -> None:
    """Set up the experiment (seeds, directories, etc.)."""
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create directories for saving results
    os.makedirs(args.save_dir, exist_ok=True)


def generate_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multimodal Gaussian data."""
    print(
        f"Generating {args.n_samples} samples from multimodal Gaussian distribution..."
    )

    data, centers = generate_multimodal_gaussian_data(
        n_samples=args.n_samples, sigma=args.sigma, random_state=args.seed
    )

    # Visualize and save the original data
    print("Visualizing data distribution...")
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    plt.scatter(centers[:, 0], centers[:, 1], color="red", s=200, marker="x")
    plt.grid(True)
    plt.title("Original Data Distribution")
    plt.savefig(os.path.join(args.save_dir, "original_data.png"))

    return data, centers


def train_model(args: argparse.Namespace, data: np.ndarray) -> DiffusionModel:
    """Train the diffusion model on the generated data."""
    print(f"Creating diffusion model with {args.time_steps} time steps...")

    # Initialize the model
    model = DiffusionModel(
        time_steps=args.time_steps,
        beta_start=1e-4,
        beta_end=0.02,
        data_dim=2,
        model_hidden_dims=[128, 256, 512, 256, 128],
    )

    # Train the model
    print(f"Training for {args.epochs} epochs...")
    loss_history = model.train(
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        verbose=True,
    )

    # Plot and save the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(args.save_dir, "training_loss.png"))

    return model


def generate_samples(
    args: argparse.Namespace,
    model: DiffusionModel,
    centers: Optional[np.ndarray] = None,
) -> None:
    """Generate samples from the trained model."""
    print(f"Generating {args.n_samples} samples from the diffusion model...")

    # Generate samples
    samples = model.sample(args.n_samples).numpy()

    # Visualize and save the generated samples
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], color="red", s=200, marker="x")

    plt.grid(True)
    plt.title("Generated Samples")
    plt.savefig(os.path.join(args.save_dir, "generated_samples.png"))

    # Save the samples to a file
    np.save(os.path.join(args.save_dir, "generated_samples.npy"), samples)


def visualize_diffusion_process(
    args: argparse.Namespace, model: DiffusionModel
) -> None:
    """Visualize the diffusion and reverse diffusion process."""
    # Get a random subset of the data
    data, centers = generate_multimodal_gaussian_data(
        n_samples=16, sigma=args.sigma, random_state=args.seed
    )

    # Convert to tensor
    data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)

    # Create a grid of plots showing the diffusion process
    plt.figure(figsize=(20, 8))

    # Original data
    plt.subplot(2, 6, 1)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.8)
    plt.title("Original Data")
    plt.grid(True)

    # Forward diffusion at different timesteps
    # Calculate 5 evenly spaced timesteps including 0 and time_steps-1
    step_size = max(1, (model.time_steps - 1) // 4)
    timesteps = [
        min(t, model.time_steps - 1) for t in range(0, model.time_steps, step_size)
    ]
    timesteps = timesteps[:5]  # Ensure we have exactly 5 timesteps

    # Pad the list if we don't have enough timesteps
    while len(timesteps) < 5:
        timesteps.append(model.time_steps - 1)

    for i, t in enumerate(timesteps):
        # Apply diffusion
        x_t, _ = model.diffuse(data_tensor, t)
        x_t = x_t.numpy()

        # Plot
        plt.subplot(2, 6, i + 2)
        plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.8)
        plt.title(f"t = {t}")
        plt.grid(True)

    # Reverse diffusion (sampling)
    noise = tf.random.normal(shape=(16, 2))
    x_t = noise

    plt.subplot(2, 6, 7)
    plt.scatter(noise[:, 0], noise[:, 1], alpha=0.8)
    plt.title("Initial Noise")
    plt.grid(True)

    # Reverse diffusion at different timesteps
    # Use the first 4 timesteps in reverse order
    reverse_timesteps = sorted(timesteps[:-1], reverse=True)

    for i, t in enumerate(reverse_timesteps):
        # Apply single step of reverse diffusion
        x_t = model._sample_step(x_t, t)

        # Plot
        plt.subplot(2, 6, i + 8)
        plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.8)
        plt.title(f"t = {t}")
        plt.grid(True)

    # Final samples
    final_samples = model._sample_step(x_t, 0).numpy()
    plt.subplot(2, 6, 12)
    plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.8)
    plt.title("Final Samples")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "diffusion_process.png"))


def main() -> None:
    """Main function to run the training script."""
    # Parse command line arguments
    args = parse_args()

    # Set up experiment (seeds, directories)
    setup_experiment(args)

    # Generate data
    data, centers = generate_data(args)

    # Train the model
    model = train_model(args, data)

    # Generate samples from the trained model
    generate_samples(args, model, centers)

    # Visualize diffusion process
    visualize_diffusion_process(args, model)

    print(f"Training completed. Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
