import numpy as np

from archive.states_list import state_3
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State
from nn_models.utils.named_tuples import *


class Vectorizer:
    def __init__(self):
        pass

    def append_tuples(self, old_tuple, new_tuples_list, seq_len):
        mask = []
        for new_tuple in new_tuples_list:
            for i in range(len(old_tuple._fields)):
              old_tuple[i].append(new_tuple[i])
            mask.append(1)
        if len(new_tuples_list) < seq_len:
            for _ in range(seq_len - len(new_tuples_list)):
                for i in range(len(old_tuple._fields)):
                    old_tuple[i].append(new_tuples_list[-1][i])
            mask.append(0)
        return mask

    def price_to_input(self, gems_collection : GemsCollection):
       return PriceTuple(*[gems_collection.gems_dict[gem_color] for gem_color in GemColor if gem_color !=
             GemColor.GOLD])

    def gems_to_input(self, gems_collection : GemsCollection, n=6):
        return GemsTuple(*[gems_collection.gems_dict[gem_color] for gem_color in GemColor])

    def card_to_input(self, card):
        profit_vec = card.discount_profit.value-1
        price_vec = self.price_to_input(card.price)
        victory_points_vec = card.victory_points
        return CardTuple(profit_vec, *list(price_vec), victory_points_vec)

    def noble_to_input(self, noble):
        price_vec = self.price_to_input(noble.price)
        return NobleTuple(*list(price_vec))

    def board_to_input(self, board: Board):
        cards_on_board = CardTuple([], [], [], [], [], [], [])
        nobles_on_board = NobleTuple([], [], [], [], [])
        cards_mask = self.append_tuples(cards_on_board, [self.card_to_input(card) for card in board.cards_on_board], 12)
        nobles_mask = self.append_tuples(nobles_on_board, [self.noble_to_input(noble) for noble in board.nobles_on_board], 3)
        list_of_args = list(self.gems_to_input(board.gems_on_board)) + list(cards_on_board) + list(nobles_on_board) + \
                       [cards_mask] + [nobles_mask]
        print(list_of_args)

        return BoardTuple(*list_of_args)

bb = state_3.board
print(Vectorizer().board_to_input(bb))

    # def players_hand_to_vectors(self, players_hand: PlayersHand):
    #     discount = self.price_to_tuple(players_hand.discount())
    #     gems_possessed = self.gems_collection_to_tuple(players_hand.gems_possessed)
    #     nobles_posessed = self.one_hot(len(players_hand.nobles_possessed), 4)
    #     cards_reserved_list = []
    #     for card in players_hand.cards_reserved:
    #         cards_reserved_list.append(self.card_to_tuple(card))
    #     cards_reserved_list = cards_reserved_list + [None]*(3 - len(cards_reserved_list))
    #     victory_points = players_hand.number_of_my_points()
    #     return PlayerTuple(discount, gems_possessed, cards_reserved_list, victory_points, nobles_posessed)
    #

    #
    #     return BoardTuple(gems_on_board, cards_on_board_list, nobles_on_board_list)
    #
    # def observation_to_vector(self, observation : DeterministicObservation):
    #
    #     local_state = observation.recreate_state()
    #     active_player = self.players_hand_to_vectors(local_state.active_players_hand())
    #     previous_player = self.players_hand_to_vectors(local_state.previous_players_hand())
    #     board_info = self.board_to_vectors(local_state.board)
    #
    #     return ObservationTuple(active_player, previous_player, board_info)