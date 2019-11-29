from gym.spaces import Space
from typing import Dict

from gym_splendor_code.envs.data.data_loader import name_to_card_dict, name_to_noble_dict
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State


class SplendorObservationSpace(Space):
    """This class contains all information we want to share with the agents playing Splendor. The difference between
    SplendorObservationSpace and State is that State contains all information about the current_state of game (including list
    of cards that are not yet revealed and class SplendorObservationSpace contains only some part of it that is
    accessible by the player. By modifying this class we can change what agent knows about the current_state of the game."""

    def __init__(self, all_cards=None, all_nobles=None):
        super().__init__()
        self.all_cards = all_cards
        self.all_nobles = all_nobles

    def __repr__(self):
        return 'Observation space in Splendor. It contains all information accessible to one player (so for example in \n' \
               'a default setting in does not contain the list of hidden cards. One observation has the following structure: \n' \
               'It is a dictionary with keys: \n' \
               '1) cards_on_board_names - a set of names of card lying on the board \n' \
               '2) gems_on_board - a collection of gems on board \n ' \
               '3) active_player_id - a number that indicates which player is active in the current current_state \n' \
               '4) players_hands - a list of dictionaries refering to consective players hands. Each dictionary in this \n' \
               'list contains the following keys:' \
               'a) cards_possessed_names - set of names of cards possesed by the players hand \n'\
               'b) cards_reserved_names - set of names of cards reserved by the players hand \n' \
               'c) gems_possessed - collection of gems possessed by the players hand'