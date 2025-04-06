import unittest
from typing import Any, List, Tuple

import numpy as np


class TestKerasInstall(unittest.TestCase):
    def test_keras_import(self) -> None:
        """Test if Keras can be imported correctly."""
        try:
            import keras

            self.assertTrue(True, "Keras imported successfully")
        except ImportError:
            self.fail("Failed to import Keras")

    def test_tensorflow_import(self) -> None:
        """Test if TensorFlow can be imported correctly."""
        try:
            import tensorflow as tf

            self.assertTrue(True, "TensorFlow imported successfully")
        except ImportError:
            self.fail("Failed to import TensorFlow")

    def test_simple_model_creation(self) -> None:
        """Test if a simple Keras model can be created."""
        try:
            import keras
            from keras import layers

            # Create a very simple model
            model = keras.Sequential(
                [
                    layers.Dense(10, activation="relu", input_shape=(5,)),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )

            # Verify model properties
            self.assertEqual(len(model.layers), 2, "Model should have 2 layers")
            self.assertEqual(
                model.input_shape, (None, 5), "Input shape should be (None, 5)"
            )
            self.assertEqual(
                model.output_shape, (None, 1), "Output shape should be (None, 1)"
            )

        except Exception as e:
            self.fail(f"Failed to create a simple Keras model: {str(e)}")


if __name__ == "__main__":
    unittest.main()
