"""Base class for trainers."""


class Trainer:
    """Base class for trainers.

    Trainer is something that can train a neural network using data from memory.
    In the most basic setup, it just samples data from a replay buffer. By
    abstracting Trainer out, we can also support other setups, e.g. tabular
    learning on a tree.
    """

    def __init__(self, input_shape):
        """No-op constructor just to specify the interface."""
        del input_shape

    def add_episode(self, episode):
        """Adds an episode to memory."""
        raise NotImplementedError

    def train_epoch(self, network):
        """Runs one epoch of training."""
        raise NotImplementedError
