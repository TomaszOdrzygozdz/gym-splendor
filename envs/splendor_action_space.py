from gym.spaces import Space

class SplendorActionSpace(Space):
    """A discrete space of actions in Splendor. Each action is an object of class Action."""

    def __init__(self):
        self.list_of_actions = []
        self.list_of_actions_gem_trade = []
        self.list_of_actions_buy = []
        self.list_of_actions_reservation = []

    def contains(self, x):
        """Chcecks if a given action belongs to the space"""

    def
