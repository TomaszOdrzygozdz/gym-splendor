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


    def state_to_observation(self, state:State) -> Dict:

        cards_on_board_names = {card.name for card in state.board.cards_on_board if card is not None}
        nobles_on_board_names = {noble.name for noble in state.board.nobles_on_board if noble is not None}
        gems_on_board = state.board.gems_on_board.__copy__()
        active_player_id = state.active_player_id
        players_hands = [{'cards_possessed_names': {card.name for card in players_hand.cards_possessed if card is not None},
                          'cards_reserved_names' : {card.name for card in players_hand.cards_reserved if card is not None},
                          'nobles_possessed_names' : {noble.name for noble in players_hand.nobles_possessed if noble is not None},
                          'gems_possessed' : players_hand.gems_possessed.__copy__()} for players_hand in state.list_of_players_hands]

        return {'cards_on_board_names' : cards_on_board_names, 'nobles_on_board_names': nobles_on_board_names,
                'gems_on_board' : gems_on_board, 'active_player_id': active_player_id, 'players_hands' : players_hands}

    def observation_to_state(self, observation) -> State:
        """Loads observation and return a current_state that agrees with the observation. Warning: this method is ambiguous,
        that is, many states can have the same observation (they may differ in the order of hidden cards)."""
        state = State(all_cards=self.all_cards, all_nobles=self.all_nobles, prepare_state=False)
        cards_on_board_names = observation['cards_on_board_names']
        nobles_on_board_names = observation['nobles_on_board_names']
        for card_name in cards_on_board_names:
                card = name_to_card_dict[card_name]
                state.board.cards_on_board.add(card)
                state.board.deck.decks_dict[card.row].remove(card)

        for noble_name in nobles_on_board_names:
            noble = name_to_noble_dict[noble_name]
            state.board.nobles_on_board.add(noble)
            state.board.deck.deck_of_nobles.remove(noble)

        state.board.gems_on_board = observation['gems_on_board']

        players_hands = []
        for player_observation in observation['players_hands']:
            players_hand = PlayersHand()
            players_hand.gems_possessed = player_observation['gems_possessed']
            for card_name in player_observation['cards_possessed_names']:
                card = name_to_card_dict[card_name]
                players_hand.cards_possessed.add(card)
                state.board.deck.decks_dict[card.row].remove(card)
            for card_name in player_observation['cards_reserved_names']:
                card = name_to_card_dict[card_name]
                players_hand.cards_reserved.add(card)
                state.board.deck.decks_dict[card.row].remove(card)
            for noble_name in player_observation['nobles_possessed_names']:
                noble = name_to_noble_dict[noble_name]
                players_hand.nobles_possessed.add(noble)
                state.board.deck.deck_of_nobles.remove(noble)
            players_hands.append(players_hand)

        state.active_player_id = observation['active_player_id']
        state.list_of_players_hands = players_hands
        return state

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