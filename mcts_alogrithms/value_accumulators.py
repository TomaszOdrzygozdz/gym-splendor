import math
import numpy as np


class ValueAccumulator:

    def __init__(self, value, state=None):
        # Creates and initializes with typical add
        self.add(value)

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


class ScalarMeanMaxValueAccumulator(ValueAccumulator):

    def __init__(self, value=0, state=None, mean_max_coeff=1.0):
        self._sum = 0.0
        self._count = 0
        self._max = value
        self.mean_max_coeff = mean_max_coeff
        self.auxiliary_loss = 0.0
        super().__init__(value, state)

    def add(self, value):
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def get(self):
        return (self._sum / self._count)*self.mean_max_coeff \
               + self._max*(1-self.mean_max_coeff)

    def index(self, parent_value=None, action=None):
        return self.get() + self.auxiliary_loss  # auxiliary_loss alters tree traversal in mcts

    def final_index(self, parent_value=None, action=None):
        return self.index(parent_value, action)

    def target(self):
        return self.get()

    def count(self):
        return self._count