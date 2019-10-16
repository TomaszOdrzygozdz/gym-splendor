from abc import abstractmethod

from envs.mechanics.card import Card
from envs.mechanics.gems_collection import GemsCollection
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

    def change_active_player(self, state: State) -> None:
        """Changes active player to the next one."""
        state.active_player_id = (state.active_player_id + 1)%len(state.list_of_players_hands)

class ActionBuyCard(Action):
    """Action of buying a card."""
    action_type = 'buy'

    def __init__(self, card: Card):
        """Parameters:
        _ _ _ _ _ _ _ _
        card: Card to buy."""
        self.card = card

    def execute(self,
                state: State) -> None:

        #First we need to find the price players has to pay for a card after considering his discount
        price_after_discount = self.card.price % state.active_players_hand().discount()
        state.active_players_hand().cards_possessed.add(self.card)
        state.active_players_hand().gems_possessed -= price_after_discount
        state.board.gems_on_board += price_after_discount

class ActionTradeGems(Action):
    """Action of trading gems with board."""
    action_type = 'trade_gems'

    def __init__(self, gems_from_board_to_player: GemsCollection):
        """Parameters:
        _ _ _ _ _ _ _ _
        gems_from_board: Gems collection describing gems that will be taken from board and added to players hand.
        Negative value means that gem (or gems) will be returned to the board.
        """
        self.gems_from_board_to_player = gems_from_board_to_player

    def execute(self,
                state: State) -> None:
        state.board.gems_on_board -= self.gems_from_board_to_player
        state.active_players_hand().gems_possessed += self.gems_from_board_to_player


class ActionReserveCard(Action):
    """Action of reserving a card."""
    action_type = 'reserve'

    def __init__(self, card: Card):
        """Parameters:
        _ _ _ _ _ _ _ _
        card: Card to reserve.
        """
        self.card = card

    def execute(self,
                state: State) -> None:

        state.board.cards_on_board.remove(self.card)
        state.active_players_hand().cards_r








