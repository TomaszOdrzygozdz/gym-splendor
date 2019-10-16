from abc import abstractmethod

from envs.mechanics.card import Card
from envs.mechanics.state import State


class Action():
    """An abstract class for any action in Splendor."""

    @property
    @abstractmethod
    #Variable storing the name of the type of action
    def action_type(self):
        pass

    @abstractmethod
    def execute(self,
                state: State) -> None:
        """Executes action on the given state."""
        pass

    @abstractmethod
    def is_legal(self,
                 state: State):
        pass

class ActionBuyCard(Action):
    """Action of buying a card."""
    action_type = 'Buy'

    def __init__(self, card: Card):
        self.card = card

    def execute(self,
                state: State) -> None:

        #First we need to find the price players has to pay for a card after considering his discount
        price_after_discount = self.card.price % state.player



