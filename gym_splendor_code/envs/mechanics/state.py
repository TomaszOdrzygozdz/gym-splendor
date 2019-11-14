from typing import Set, List

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.noble import Noble
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.game_settings import INITIAL_GEMS_ON_BOARD_DICT
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.data.data_loader import load_all_cards
from gym_splendor_code.envs.data.data_loader import load_all_nobles
import simplejson as json


class State():
    """This class keeps all information about the current_state of the game."""

    def __init__(self,
                 list_of_players_hands: List = None,
                 all_cards: Set[Card] = None,
                 all_nobles: Set[Noble] = None,
                 gems_on_board: GemsCollection = None,
                 prepare=True) -> None:

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

    def prepare_state(self):

        self.active_player_id = 0  # index
        self.board.deck.shuffle()
        self.board.lay_cards_on_board()
        self.board.lay_nobles_on_board()
    #
    #     else:
    #         if file:
    #             with open(from_state) as json_data:
    #                 vector = json.load(json_data)
    #                 json_data.close()
    #                 vector = eval(vector.replace("NULL", "set()"))
    #         else:
    #             pass
    #             #vector = eval(from_state.replace("NULL", "set()"))
    #         vector = from_state
    #         self.active_player_id = vector['active_player_id']
    #         self.list_of_players_hands[self.active_player_id].from_json(vector['active_player_hand'])
    #         self.list_of_players_hands[(self.active_player_id - 1) % len(self.list_of_players_hands)].from_json(
    #             vector['previous_player_hand'])
    #         self.board.from_json(vector)
    #
    #         # Adding nobles
    #         for i in vector['active_player_hand']['noble_possessed_ids']:
    #             self.list_of_players_hands[self.active_player_id].nobles_possessed.add(
    #                 self.board.deck.pop_noble_by_id(i))
    #
    #         for i in vector['previous_player_hand']['noble_possessed_ids']:
    #             self.list_of_players_hands[
    #                 (self.active_player_id - 1) % len(self.list_of_players_hands)].nobles_possessed.add(
    #                 self.board.deck.pop_noble_by_id(i))
    #
    #         for i in vector['board']['nobles_on_board']:
    #             self.board.nobles_on_board.add(self.board.deck.pop_noble_by_id(i))
    #
    #         # Adding cards
    #         for i in vector['active_player_hand']['cards_possessed_ids']:
    #             self.list_of_players_hands[self.active_player_id].cards_possessed.add(self.board.deck.pop_card_by_id(i))
    #
    #         for i in vector['active_player_hand']['cards_reserved_ids']:
    #             self.list_of_players_hands[self.active_player_id].cards_reserved.add(self.board.deck.pop_card_by_id(i))
    #
    #         for i in vector['previous_player_hand']['cards_possessed_ids']:
    #             self.list_of_players_hands[
    #                 (self.active_player_id - 1) % len(self.list_of_players_hands)].cards_possessed.add(
    #                 self.board.deck.pop_card_by_id(i))
    #
    #         for i in vector['previous_player_hand']['cards_reserved_ids']:
    #             self.list_of_players_hands[(self.active_player_id - 1)%len(self.list_of_players_hands)].cards_reserved.add(self.board.deck.pop_card_by_id(i))
    #
    #         if ordered_deck:
    #             self.board.deck.order_deck(vector)
    #         else:
    #             self.board.deck.shuffle()

    def active_players_hand(self):
        """Returns the hand of active player"""
        return self.list_of_players_hands[self.active_player_id]


    def previous_players_hand(self):

        """Return the hans of the previous player"""
        return self.list_of_players_hands[(self.active_player_id - 1) % len(self.list_of_players_hands)]

    def to_dict(self):
        return {'active_player_hand': self.active_players_hand().jsonize(),
                'previous_player_hand': self.previous_players_hand().jsonize(),
                'board': self.board.jsonize(),
                'active_player_id': self.active_player_id}

    def load_from_dict(self, state_as_dict, order_deck = False):
        self.active_player_id = state_as_dict['active_player_id']
        self.list_of_players_hands[self.active_player_id].from_json(state_as_dict['active_player_hand'])
        self.list_of_players_hands[(self.active_player_id - 1) % len(self.list_of_players_hands)].from_json(
            state_as_dict['previous_player_hand'])
        self.board.from_json(state_as_dict)

        # Adding nobles
        for i in state_as_dict['active_player_hand']['noble_possessed_ids']:
            self.list_of_players_hands[self.active_player_id].nobles_possessed.add(
                self.board.deck.pop_noble_by_id(i))

        for i in state_as_dict['previous_player_hand']['noble_possessed_ids']:
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

        for i in state_as_dict['previous_player_hand']['cards_possessed_ids']:
            self.list_of_players_hands[
                (self.active_player_id - 1) % len(self.list_of_players_hands)].cards_possessed.add(
                self.board.deck.pop_card_by_id(i))

        for i in state_as_dict['previous_player_hand']['cards_reserved_ids']:
            self.list_of_players_hands[
                (self.active_player_id - 1) % len(self.list_of_players_hands)].cards_reserved.add(
                self.board.deck.pop_card_by_id(i))

        if order_deck:
            self.board.deck.order_deck(state_as_dict)
        else:
            self.board.deck.shuffle()

