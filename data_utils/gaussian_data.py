"""Multimodal Gaussian data generation utilities."""

from typing import Optional, Tuple

import numpy as np


def generate_multimodal_gaussian_data(
    n_samples: int,
    sigma: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data from a mixture of 4 Gaussian distributions in 2D.

    Data is sampled equally from 4 Gaussian distributions centered at:
    - (1, 0) with sigma
    - (0, 1) with sigma
    - (-1, 0) with sigma
    - (0, -1) with sigma

    Args:
        n_samples: Number of samples to generate
        sigma: Standard deviation for each Gaussian
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (data, centers) where:
        - data: numpy array of shape (n_samples, 2)
        - centers: numpy array of shape (4, 2) with the centers of each Gaussian
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Define the centers of the four Gaussian distributions
    centers = np.array(
        [
            [1.0, 0.0],  # Right
            [0.0, 1.0],  # Top
            [-1.0, 0.0],  # Left
            [0.0, -1.0],  # Bottom
        ]
    )

    samples_per_mode = n_samples // 4
    data = np.zeros((n_samples, 2))

    for i in range(4):
        start_idx = i * samples_per_mode
        end_idx = start_idx + samples_per_mode if i < 3 else n_samples
        n_mode_samples = end_idx - start_idx

        # Generate samples for this mode
        mode_samples = np.random.normal(
            loc=centers[i], scale=sigma, size=(n_mode_samples, 2)
        )

        data[start_idx:end_idx] = mode_samples

    # Shuffle the data
    np.random.shuffle(data)

    return data, centers


def visualize_data(
    data: np.ndarray,
    centers: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the 2D multimodal Gaussian data.

    Args:
        data: numpy array of shape (n_samples, 2)
        centers: numpy array of shape (n_centers, 2) with centers
        save_path: If provided, save the figure to this path
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], color="red", s=200, marker="x")

    plt.grid(True)
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    plt.title("Multimodal Gaussian Distribution")
    plt.xlabel("x")
    plt.ylabel("y")

    if save_path:
        plt.savefig(save_path)

    plt.show()
