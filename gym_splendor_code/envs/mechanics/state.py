from typing import Set, List, Dict

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.noble import Noble
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.game_settings import INITIAL_GEMS_ON_BOARD_DICT
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.data.data_loader import load_all_cards
from gym_splendor_code.envs.data.data_loader import load_all_nobles
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


class State():
    """This class keeps all information about the current_state of the game."""

    def __init__(self,
                 list_of_players_hands: List = None,
                 all_cards: Set[Card] = None,
                 all_nobles: Set[Noble] = None,
                 gems_on_board: GemsCollection = None,
                 load_from_state_as_dict:StateAsDict = None,
                 prepare_state: bool = False) -> None:

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
        self.active_player_id = 0

        if load_from_state_as_dict is None and prepare_state:
            self.prepare_state()
        elif load_from_state_as_dict is not None:
            self.load_from_dict(load_from_state_as_dict)


    def prepare_state(self):

        self.active_player_id = 0  # index
        self.board.deck.shuffle()
        self.board.lay_cards_on_board()
        self.board.lay_nobles_on_board()

    def active_players_hand(self):
        """Returns the hand of active player"""
        return self.list_of_players_hands[self.active_player_id]


    def previous_players_hand(self):

        """Return the hans of the previous player"""
        return self.list_of_players_hands[(self.active_player_id - 1) % len(self.list_of_players_hands)]

    def to_dict(self) -> StateAsDict:
        return StateAsDict({'active_player_hand': self.active_players_hand().jsonize(),
                'other_player_hand': self.previous_players_hand().jsonize(),
                'board': self.board.jsonize(),
                'active_player_id': self.active_player_id})

    def load_from_dict(self, state_as_dict: StateAsDict, order_deck = False):
        self.active_player_id = state_as_dict['active_player_id']
        self.list_of_players_hands[self.active_player_id].from_json(state_as_dict['active_player_hand'])
        self.list_of_players_hands[(self.active_player_id - 1) % len(self.list_of_players_hands)].from_json(
            state_as_dict['other_player_hand'])
        self.board.from_json(state_as_dict)

        # Adding nobles
        for i in state_as_dict['active_player_hand']['noble_possessed_ids']:
            self.list_of_players_hands[self.active_player_id].nobles_possessed.add(
                self.board.deck.pop_noble_by_id(i))

        for i in state_as_dict['other_player_hand']['noble_possessed_ids']:
            self.list_of_players_hands[
                (self.active_player_id - 1) % len(self.list_of_players_hands)].nobles_possessed.add(
                self.board.deck.pop_noble_by_id(i))

        for i in state_as_dict['board']['nobles_on_board']:
            self.board.nobles_on_board.add(self.board.deck.pop_noble_by_id(i))

        # Adding cards
        for i in state_as_dict['active_player_hand']['cards_possessed_ids']:
            self.list_of_players_hands[self.active_player_id].cards_possessed.add(self.board.deck.pop_card_by_id(i))

        for i in state_as_dict['active_player_hand']['cards_reserved_ids']:
            self.list_of_players_hands[self.active_player_id].cards_reserved.add(self.board.deck.pop_card_by_id(i))

        for i in state_as_dict['other_player_hand']['cards_possessed_ids']:
            self.list_of_players_hands[
                (self.active_player_id - 1) % len(self.list_of_players_hands)].cards_possessed.add(
                self.board.deck.pop_card_by_id(i))

        for i in state_as_dict['other_player_hand']['cards_reserved_ids']:
            self.list_of_players_hands[
                (self.active_player_id - 1) % len(self.list_of_players_hands)].cards_reserved.add(
                self.board.deck.pop_card_by_id(i))

        if order_deck:
            self.board.deck.order_deck(state_as_dict)
        else:
            self.board.deck.shuffle()