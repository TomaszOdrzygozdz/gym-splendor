"""Utilities for gym_open_ai spaces."""

import gym_open_ai


def space_iter(action_space):
    """Returns an iterator over points in a gym_open_ai space."""
    try:
        return iter(action_space)
    except TypeError:
        if isinstance(action_space, gym_open_ai.spaces.Discrete):
            return iter(range(action_space.n))
        else:
            raise TypeError('Space {} does not support iteration.'.format(
                type(action_space)
            ))
