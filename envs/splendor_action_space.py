from gym.spaces import Space

from envs.mechanics.action_space_generator import generate_all_legal_trades, generate_all_legal_buys, \
    generate_all_legal_reservations
from envs.mechanics.state import State


class SplendorActionSpace(Space):
    """A discrete space of actions in Splendor. Each action is an object of class Action."""

    def __init__(self):
        super().__init__()
        self.list_of_actions = []
        self.list_of_actions_gem_trade = []
        self.list_of_actions_buy = []
        self.list_of_actions_reservation = []

    def update(self,
               state: State) -> None:

        self.list_of_actions_gem_trade = generate_all_legal_trades(state)
        self.list_of_actions_buy = generate_all_legal_buys(state)
        self.list_of_actions_reservation = generate_all_legal_reservations(state)
        self.list_of_actions = self.list_of_actions_gem_trade + self.list_of_actions_buy \
                               + self.list_of_actions_reservation
        self.shape = (len(self.list_of_actions))

    def contains(self, x):
        """Chcecks if a given action belongs to the space"""
        return x in self.list_of_actions

    def sample(self):
        assert len(self.list_of_actions) > 0, 'No actions to sample. Make sure actions space was updated.'

