import math
import numpy as np


class ValueAccumulator:

    def __init__(self, value, state=None):
        # Creates and initializes with typical add
        pass

    def add(self, value):
        """Adds an abstract value to the accumulator.

        Args:
            value: Abstract value to add.
        """
        raise NotImplementedError

    def add_auxiliary(self, value):
        """
        Additional value for traversals
        """
        raise NotImplementedError

    def get(self):
        """Returns the accumulated abstract value for backpropagation.
        """
        raise NotImplementedError

    def index(self, parent_value, action):
        """Returns an index for selecting the best node."""
        raise NotImplementedError

    def target(self):
        """Returns a target for value function training."""
        raise NotImplementedError

    def count(self):
        """Returns the number of accumulated values."""
        raise NotImplementedError

    def set_constant_value_for_terminal_node(self, perfect_value):
        self.perfect_value = perfect_value
        self._count = 1


