"""Utilities for gym spaces."""

import gym


def space_iter(action_space):
    """Returns an iterator over points in a gym space."""
    try:
        return iter(action_space)
    except TypeError:
        if isinstance(action_space, gym.spaces.Discrete):
            return iter(range(action_space.n))
        else:
            raise TypeError('Space {} does not support iteration.'.format(
                type(action_space)
            ))
