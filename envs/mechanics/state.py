from typing import Set, List

from envs.mechanics.card import Card
from envs.mechanics.gems_collection import GemsCollection
from envs.mechanics.noble import Noble
from envs.mechanics.players_hand import PlayersHand
from envs.mechanics.board import Board
from envs.utils.data_loader import load_all_cards
from envs.utils.data_loader import load_all_nobles


class State():
    """This class keeps all informations about the state of the game."""

    def __init__(self,
                 list_of_players_hands: List = None,
                 set_of_cards: Set[Card] = None,
                 set_of_nobles: Set[Noble] = None,
                 gems_collection_on_board : GemsCollection = None) -> None:

        if set_of_cards is None:
            set_of_cards = load_all_cards()
        if set_of_nobles is None:
            set_of_nobles = load_all_nobles()
        

        self.board = Board()