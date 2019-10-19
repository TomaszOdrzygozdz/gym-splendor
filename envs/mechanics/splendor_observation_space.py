import random
from gym.spaces import Space

from envs.mechanics.state import State

class SplendorObservationSpace(Space):
    """This is the state of game."""

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Observation is an object of class State. This class contains all information about the state of the ' \
               'game.'
