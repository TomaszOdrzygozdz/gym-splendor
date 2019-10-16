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

    def give_nobles(self, state: State) -> None:
        """Checks if the active player deserves to obtain noble card (or cards)."""
        for noble in state.board.nobles_on_board:
            if noble.price <= state.active_players_hand().discount():
                state.active_players_hand().nobles_possessed.add(noble)
                state.board.nobles_on_board.remove(noble)
        
class ActionBuyCard(Action):
    """Action of buying a card."""
    action_type = 'Buy'

    def __init__(self, card: Card):
        self.card = card

    def execute(self,
                state: State) -> None:

        #First we need to find the price players has to pay for a card after considering his discount
        price_after_discount = self.card.price % state.active_players_hand().discount()
        state.active_players_hand().cards_possessed.add(self.card)
        state.active_players_hand().gems_possessed -= price_after_discount
        state.board.gems_on_board += price_after_discount






