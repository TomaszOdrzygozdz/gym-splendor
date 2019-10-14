from typing import Set, List

from envs.mechanics.card import Card
from envs.mechanics.gems_collection import GemsCollection
from envs.mechanics.noble import Noble
from envs.mechanics.players_hand import PlayersHand
from envs.data.game_settings import INITIAL_GEMS_ON_BOARD_DICT
from envs.mechanics.board import Board
from envs.data.data_loader import load_all_cards
from envs.data.data_loader import load_all_nobles


class State():
    """This class keeps all informations about the state of the game."""

    def __init__(self,
                 list_of_players_hands: List = None,
                 all_cards: Set[Card] = None,
                 all_nobles: Set[Noble] = None,
                 gems_on_board : GemsCollection = None) -> None:

        if all_cards is None:
            all_cards = load_all_cards()
        if all_nobles is None:
            all_nobles = load_all_nobles()
        if gems_on_board is None:
            gems_on_board = GemsCollection(INITIAL_GEMS_ON_BOARD_DICT)
        if list_of_players_hands is None:
            list_of_players_hands = [PlayersHand("Player A"), PlayersHand("Player B")]

        self.board = Board(all_cards, all_nobles, gems_on_board)

        self.board.lay_cards_on_board()
        self.board.lay_nobles_on_board()


stanek = State()

print(len(stanek.board.cards_on_board))
print(len(stanek.board.nobles_on_board))