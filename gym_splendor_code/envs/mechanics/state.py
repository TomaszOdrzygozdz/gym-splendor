from typing import Set, List

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.noble import Noble
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.game_settings import INITIAL_GEMS_ON_BOARD_DICT
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.data.data_loader import load_all_cards
from gym_splendor_code.envs.data.data_loader import load_all_nobles


class State():
    """This class keeps all information about the state of the game."""

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

        self.list_of_players_hands = list_of_players_hands
        self.board = Board(all_cards, all_nobles, gems_on_board)

        self.active_player_id = 0 #index

        self.board.lay_cards_on_board()
        self.board.lay_nobles_on_board()

    def active_players_hand(self):
        """Returns the hand of active player"""
        return self.list_of_players_hands[self.active_player_id]

    def previous_players_hans(self):
        """Return the hans of the previous player"""
        return self.list_of_players_hands[(self.active_player_id - 1)%len(self.list_of_players_hands)]

    def vectorize(self):
        return [{'active_player_hand': self.active_players_hand().vectorize(),
                 'previous_player_hand': self.previous_players_hand().vectorize(),
                 'board': self.board(),
                 'active_player_id': self.active_player_id}]
