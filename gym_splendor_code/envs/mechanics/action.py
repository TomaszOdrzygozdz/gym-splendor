from abc import abstractmethod

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.state import State


class Action():
    """An abstract class for any action in Splendor."""

    @property
    @abstractmethod
    #Variable storing the name of the type of action
    def action_type(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def execute(self,
                state: State) -> None:
        """Executes action on the given state."""
        pass

    def give_nobles(self, state: State) -> None:
        """Checks if the active player deserves to obtain noble card (or cards)."""
        nobles_to_transfer = set()
        for noble in state.board.nobles_on_board:
            if noble.price <= state.active_players_hand().discount():
                nobles_to_transfer.add(noble)
        for noble in nobles_to_transfer:
                state.active_players_hand().nobles_possessed.add(noble)
                state.board.nobles_on_board.remove(noble)

    def change_active_player(self, state: State) -> None:
        """Changes active player to the next one."""
        state.active_player_id = (state.active_player_id + 1)%len(state.list_of_players_hands)

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
        state.board.gems_on_board = state.board.gems_on_board - self.gems_from_board_to_player
        state.active_players_hand().gems_possessed = state.active_players_hand().gems_possessed \
                                                     + self.gems_from_board_to_player
        self.change_active_player(state)

    def __eq__(self, other):
        condition1 = self.action_type == other.action_type
        if condition1:
            condition2 = self.gems_from_board_to_player == other.gems_from_board_to_player
            return condition2
        else:
            return False

    def __repr__(self):
        return 'Trade gems ' + self.gems_from_board_to_player.__repr__()


class ActionBuyCard(Action):
    """Action of buying a card."""
    action_type = 'buy'

    def __init__(self,
                 card: Card,
                 n_gold_gems_to_use: int=0,
                 use_gold_as: GemsCollection = GemsCollection()):
        """Parameters:
        _ _ _ _ _ _ _ _
        card: Card to buy.
        gold_gems_to_use: Integer determining how many golden gems will be used to pay for the card.
        use_gold_as: Gems collection that reduces the price (its sum must equal n_gold_gems_to_use)."""
        self.card = card
        assert n_gold_gems_to_use == use_gold_as.sum(), 'n_gold_gems_to_use must be equal the sum of gems in use_gold_as'
        self.n_gold_gems_to_use = n_gold_gems_to_use
        self.use_gold_as = use_gold_as


    def execute(self,
                state: State) -> None:

        #First we need to find the price players has to pay for a card after considering his discount
        price_after_discount = self.card.price % state.active_players_hand().discount()
        if self.n_gold_gems_to_use > 0:
            #take golden gems from player:
            state.active_players_hand().gems_possessed.gems_dict[GemColor.GOLD] -= self.n_gold_gems_to_use
            #reduce the price of card:
            price_after_discount -= self.use_gold_as

        state.active_players_hand().cards_possessed.add(self.card)
        if self.card in state.board.cards_on_board:
            state.board.remove_card_from_board_and_refill(self.card)
        if self.card in state.active_players_hand().cards_reserved:
            state.active_players_hand().cards_reserved.remove(self.card)
        state.active_players_hand().gems_possessed = state.active_players_hand().gems_possessed - price_after_discount
        state.board.gems_on_board = state.board.gems_on_board + price_after_discount
        self.give_nobles(state)
        self.change_active_player(state)

    def __eq__(self, other):
        condition_1 = self.action_type == other.action_type
        if condition_1:
            condition_2 = self.card == other.card
            condition_3 = self.n_gold_gems_to_use == other.n_gold_gems_to_use
            condition_4 = self.use_gold_as == other.use_gold_as
            return condition_2 and condition_3 and condition_4
        else:
            return False


    def __repr__(self):
        return 'Buy ' + self.card.__repr__() + '\n gold gems to use: {}, use gold gems as: {}'.format(self.n_gold_gems_to_use,
                                                                                                     self.use_gold_as.__repr__())

class ActionReserveCard(Action):
    """Action of reserving a card."""
    action_type = 'reserve'

    def __init__(self,
                 card: Card,
                 take_golden_gem: bool,
                 return_gem_color: GemColor = None):
        """Parameters:
        _ _ _ _ _ _ _ _
        card: Card to reserve.
        take_golden_gem: Determines if a golden gem will be given to the players hand (it may be not given
        if the player has already maximum number of gems).
        return_gem: If player has maximum number of gems, may take one more golden gem but has to return one of other
        gems to the board.
        """
        self.card = card
        self.take_golden_gem = take_golden_gem
        self.return_gem_color = return_gem_color

    def execute(self,
                state: State) -> None:
        state.board.remove_card_from_board_and_refill(self.card)
        state.active_players_hand().cards_reserved.add(self.card)
        if self.take_golden_gem:
            state.active_players_hand().gems_possessed.gems_dict[GemColor.GOLD] += 1
            state.board.gems_on_board.gems_dict[GemColor.GOLD] -= 1
            if self.return_gem_color is not None:
                state.active_players_hand().gems_possessed[self.return_gem_color] -= 1
                state.board.gems_on_board.gems_dict[self.return_gem_color] += 1
        self.change_active_player(state)

    def __eq__(self, other):
        condition_1 = self.action_type == other.action_type
        if condition_1:
            condition_2 = self.card == other.card
            condition_3 = self.take_golden_gem == other.take_golden_gem
            condition_4 = self.return_gem_color == other.return_gem_color
            return condition_2 and condition_3 and condition_4
        else:
            return False

    def __repr__(self):
        return 'Reserve ' + self.card.__repr__() + '\n take golden gem: {}, return_gem_color {}'.format(self.take_golden_gem,
                                                                                                      self.return_gem_color)