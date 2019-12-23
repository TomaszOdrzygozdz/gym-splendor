"""Deep learning framework-agnostic interface for neural networks."""


class Network:
    """Base class for networks."""

    def __init__(self, input_shape):
        """Initializes Network.

        Args:
            input_shape (tuple): Input shape.
        """
        self._input_shape = input_shape

    def train(self, data_stream):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of batches to run
                the updates on.

        Returns:
            dict: Collected metrics, indexed by name.
        """
        raise NotImplementedError

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs (Agent-dependent): Batch of inputs to run prediction on.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Returns network parameters."""
        raise NotImplementedError

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""
        raise NotImplementedError

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""
        raise NotImplementedError

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""
        raise NotImplementedError


class DummyNetwork(Network):
    """Dummy Network for testing."""

    def train(self, data_stream):
        del data_stream
        return {}

    def predict(self, inputs):
        return inputs

    @property
    def params(self):
        return None

    @params.setter
    def params(self, new_params):
        del new_params

    def save(self, checkpoint_path):
        del checkpoint_path

    def restore(self, checkpoint_path):
        del checkpoint_path
