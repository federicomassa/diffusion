"""Basic diffusion model implementation in Keras."""

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional


class TimeEmbedding(tf.keras.layers.Layer):
    """Layer to convert time steps to embeddings."""
    
    def __init__(self, embedding_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.dense = tf.keras.layers.Dense(embedding_dim, activation='swish')
        
    def build(self, input_shape):
        # Create log-spaced frequencies
        emb = tf.math.log(10000.0) / (self.half_dim - 1)
        emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -emb)
        self.emb = emb
        super().build(input_shape)
        
    def call(self, inputs):
        # Apply sin and cos embedding
        t = tf.cast(inputs, dtype=tf.float32)
        emb = tf.expand_dims(self.emb, 0)  # [1, half_dim]
        emb = t * emb  # [batch, half_dim]
        
        # Apply sin and cos
        emb_sin = tf.sin(emb)
        emb_cos = tf.cos(emb)
        
        # Concatenate and transform
        emb = tf.concat([emb_sin, emb_cos], axis=-1)
        emb = self.dense(emb)
        
        return emb


class DiffusionModel:
    """
    Basic implementation of a diffusion model for 2D data.
    
    This implements a simple DDPM (Denoising Diffusion Probabilistic Model) as 
    described in "Denoising Diffusion Probabilistic Models" by Ho et al.
    """
    
    def __init__(
        self,
        time_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        data_dim: int = 2,
        model_hidden_dims: List[int] = [128, 256, 128],
    ) -> None:
        """
        Initialize the diffusion model.
        
        Args:
            time_steps: Number of diffusion steps
            beta_start: Starting noise schedule value
            beta_end: Ending noise schedule value
            data_dim: Dimensionality of the data (default: 2 for 2D points)
            model_hidden_dims: Hidden dimensions for the neural network
        """
        self.time_steps = time_steps
        self.data_dim = data_dim
        
        # Define beta schedule and derived values
        self.betas = self._get_beta_schedule(beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
        # Create the denoising network (U-Net like architecture for 2D data)
        self.model = self._build_model(model_hidden_dims)
    
    def _get_beta_schedule(self, beta_start: float, beta_end: float) -> np.ndarray:
        """
        Generate the noise schedule for the diffusion process.
        
        Args:
            beta_start: Start value for the noise schedule
            beta_end: End value for the noise schedule
            
        Returns:
            Noise schedule as a numpy array of shape (time_steps,)
        """
        return np.linspace(beta_start, beta_end, self.time_steps)
    
    def _build_model(self, hidden_dims: List[int]) -> tf.keras.Model:
        """
        Build the neural network for denoising.
        
        Args:
            hidden_dims: List of hidden dimensions for the network
            
        Returns:
            Keras model that predicts noise given noisy data and timestep
        """
        # Input for noisy data
        x_input = tf.keras.layers.Input(shape=(self.data_dim,))
        
        # Input for timestep embedding
        time_input = tf.keras.layers.Input(shape=(1,))
        
        # Time embedding (similar to positional encoding)
        time_embedding = TimeEmbedding()(time_input)
        
        # Initial feature extraction
        h = tf.keras.layers.Dense(hidden_dims[0], activation='swish')(x_input)
        
        # Inject time embedding
        h = tf.keras.layers.Concatenate()([h, time_embedding])
        
        # Middle layers
        for dim in hidden_dims[1:]:
            h = tf.keras.layers.Dense(dim, activation='swish')(h)
        
        # Output layer (predict the noise)
        output = tf.keras.layers.Dense(self.data_dim)(h)
        
        return tf.keras.Model(inputs=[x_input, time_input], outputs=output)
    
    def diffuse(self, x_0: tf.Tensor, t: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply t steps of diffusion to the input data.
        
        Args:
            x_0: Initial clean data of shape (batch_size, data_dim)
            t: Diffusion time step
            
        Returns:
            Tuple of (noisy_data, noise) where both have shape (batch_size, data_dim)
        """
        alpha_cumprod_t = tf.constant(self.alphas_cumprod[t], dtype=tf.float32)
        
        # Generate Gaussian noise
        noise = tf.random.normal(shape=x_0.shape)
        
        # Add noise according to the diffusion process
        x_t = tf.sqrt(alpha_cumprod_t) * x_0 + tf.sqrt(1.0 - alpha_cumprod_t) * noise
        
        return x_t, noise
    
    def sample(self, batch_size: int) -> tf.Tensor:
        """
        Sample new points by running the reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Generated samples of shape (batch_size, data_dim)
        """
        # Start with pure noise
        x_T = tf.random.normal(shape=(batch_size, self.data_dim))
        x_t = x_T
        
        # Iterate backwards through diffusion steps
        for t in range(self.time_steps - 1, -1, -1):
            x_t = self._sample_step(x_t, t)
        
        return x_t
    
    def _sample_step(self, x_t: tf.Tensor, t: int) -> tf.Tensor:
        """
        Perform a single reverse diffusion step.
        
        Args:
            x_t: Current noisy samples of shape (batch_size, data_dim)
            t: Current time step
            
        Returns:
            Denoised samples for the previous timestep
        """
        batch_size = tf.shape(x_t)[0]
        
        # Time step tensor
        t_tensor = tf.ones((batch_size, 1), dtype=tf.int32) * t
        
        # Predict noise
        predicted_noise = self.model([x_t, t_tensor], training=False)
        
        # Get the coefficients for the mean
        alpha_t = tf.constant(self.alphas[t], dtype=tf.float32)
        alpha_cumprod_t = tf.constant(self.alphas_cumprod[t], dtype=tf.float32)
        beta_t = tf.constant(self.betas[t], dtype=tf.float32)
        
        # Previous alpha_cumprod
        alpha_cumprod_prev = tf.constant(
            self.alphas_cumprod[t-1] if t > 0 else 1.0, 
            dtype=tf.float32
        )
        
        # Variance
        variance = tf.constant(0.0, dtype=tf.float32)
        if t > 0:
            variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
        
        # Calculate mean
        mean_coef1 = 1.0 / tf.sqrt(alpha_t)
        mean_coef2 = (1.0 - alpha_t) / tf.sqrt(1.0 - alpha_cumprod_t)
        
        mean = mean_coef1 * (x_t - mean_coef2 * predicted_noise)
        
        # Sample
        if t > 0:
            noise = tf.random.normal(shape=x_t.shape)
            return mean + tf.sqrt(variance) * noise
        else:
            return mean
    
    def train_step(
        self,
        x_0: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            x_0: Clean data of shape (batch_size, data_dim)
            optimizer: Keras optimizer
            
        Returns:
            Dictionary with the loss value
        """
        batch_size = tf.shape(x_0)[0]
        
        # Sample random timesteps
        t = tf.random.uniform(
            shape=(batch_size, 1),
            minval=0,
            maxval=self.time_steps,
            dtype=tf.int32
        )
        
        with tf.GradientTape() as tape:
            # Diffuse the data
            alphas_cumprod_t = tf.gather(
                tf.constant(self.alphas_cumprod, dtype=tf.float32),
                t
            )
            
            # Generate noise
            noise = tf.random.normal(shape=x_0.shape)
            
            # Add noise to the data
            x_t = tf.sqrt(alphas_cumprod_t) * x_0 + tf.sqrt(1.0 - alphas_cumprod_t) * noise
            
            # Predict noise
            predicted_noise = self.model([x_t, t], training=True)
            
            # Calculate MSE loss
            loss = tf.reduce_mean(tf.square(noise - predicted_noise))
        
        # Update model weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return {"loss": loss}
    
    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the diffusion model on data.
        
        Args:
            data: Training data of shape (n_samples, data_dim)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for the Adam optimizer
            verbose: Whether to print progress
            
        Returns:
            List of loss values per epoch
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        n_samples = data.shape[0]
        steps_per_epoch = n_samples // batch_size
        
        loss_history = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            data_shuffled = data[indices]
            
            epoch_loss = 0.0
            
            for step in range(steps_per_epoch):
                # Get batch
                start_idx = step * batch_size
                end_idx = start_idx + batch_size
                x_batch = data_shuffled[start_idx:end_idx]
                
                # Train step
                step_result = self.train_step(tf.convert_to_tensor(x_batch, dtype=tf.float32), optimizer)
                epoch_loss += step_result["loss"]
            
            # Average loss
            epoch_loss /= steps_per_epoch
            loss_history.append(epoch_loss.numpy())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        return loss_history 