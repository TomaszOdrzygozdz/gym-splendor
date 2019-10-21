import random
from gym.spaces import Space

from envs.mechanics.gems_collection import GemsCollection
from envs.mechanics.state import State

class SplendorObservationSpace(Space):
    """This class contains all information we want to share with the agents playing Splendor. The difference between
    SplendorObservationSpace and State is that State contains all information about the state of game (including list
    of cards that are not yet revealed and class SplendorObservationSpace contains only some part of it that is
    accessible by the player. By modifying this class we can change what agent knows about the state of the game."""

    def __init__(self):
        super().__init__()

    def state_to_observation(self, state):
        cards_on_board_names = {card.name for card in state.board.cards_on_board}
        gems_on_board = state.board.gems_on_board.__copy__()
        active_player_id = state.active_player_id
        players_hands = [{'cards_possessed_names': {card.name for card in players_hand.cards_possessed},
                          'cards_reserved_names' : {card.name for card in players_hand.cards_reserved},
                          'gems_possessed_names' : players_hand.gems_possessed.__copy__()} for players_hand in state.list_of_players_hands]

        return {'cards_on_board_names' : cards_on_board_names, 'gems_on_board' : gems_on_board,
                'active_player_id' : active_player_id, 'players_hands' : players_hands}

    def __repr__(self):
        return 'Observation space in Splendor. It contains all information accessible to one player (so for example in \n' \
               'a default setting in does not contain the list of hidden cards. One observation has the following structure: \n' \
               'It is a dictionary with keys: \n' \
               '1) cards_on_board_names - a set of names of card lying on the board \n' \
               '2) gems_on_board - a collection of gems on board \n ' \
               '3) active_player_id - a number that indicates which player is active in the current state \n' \
               '4) players_hands - a list of dictionaries refering to consective players hands. Each dictionary in this \n' \
               'list contains the following keys:' \
               'a) cards_possessed_names - set of names of cards possesed by the players hand \n'\
               'b) cards_reserved_names - set of names of cards reserved by the players hand \n' \
               'c) gems_possessed - collection of gems possessed by the players hand'