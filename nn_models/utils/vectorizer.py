import numpy as np

from gym_splendor_code.envs.mechanics.board import Board
from typing import List
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State
from nn_models.utils.named_tuples import *


class Vectorizer:
    def __init__(self):
        pass

    def board_to_tensors(self, board: Board):
        board_tuple = self.board_to_input(board)
        list_of_tensors = []
        for i in range(6):
            list_of_tensors.append(np.array(board_tuple[i]).reshape(1,1))
        for i in range(6, 13):
            list_of_tensors.append(np.array(board_tuple[i]).reshape(1,12))
        for i in range(13, 18):
            list_of_tensors.append(np.array(board_tuple[i]).reshape(1,3))
        list_of_tensors.append(np.array(board_tuple[18]).reshape(1,12))
        list_of_tensors.append(np.array(board_tuple[19]).reshape(1, 3))
        return list_of_tensors

    # def player_to_tensor(self, players_hans: PlayersHand):
    #     player

    def append_tuples(self, old_tuple, new_tuples_list, seq_len, false_elem):
        mask = []
        for new_tuple in new_tuples_list:
            for i in range(len(old_tuple._fields)):
              old_tuple[i].append(new_tuple[i])
            mask.append(1)
        if len(new_tuples_list) < seq_len:
            for _ in range(seq_len - len(new_tuples_list)):
                for i in range(len(old_tuple._fields)):
                    old_tuple[i].append(false_elem[i])
                mask.append(0)
        if len(new_tuples_list) == 0:
            mask = [1]*seq_len
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
        cards_mask = self.append_tuples(cards_on_board, [self.card_to_input(card) for card in board.cards_on_board], 12, FALSE_CARD)
        nobles_mask = self.append_tuples(nobles_on_board, [self.noble_to_input(noble) for noble in board.nobles_on_board], 3, FALSE_NOBLE)
        list_of_tensors = []
        for x in self.gems_to_input(board.gems_on_board):
            list_of_tensors.append(np.array(x).reshape(1, 1))
        for x in cards_on_board:
            list_of_tensors.append(np.array(x).reshape(1,12))
        for x in nobles_on_board:
            list_of_tensors.append(np.array(x).reshape(1,3))
        list_of_tensors.append(np.array(cards_mask).reshape(1,12))
        list_of_tensors.append(np.array(nobles_mask).reshape(1, 3))

        #list_of_args = list(self.gems_to_input(board.gems_on_board)) + list(cards_on_board) + list(nobles_on_board) + \
          #             [cards_mask] + [nobles_mask]
        #return BoardTuple(*list_of_args)
        return list_of_tensors

    def players_hand_to_input(self, players_hand: PlayersHand):
        reserved_cards_list = CardTuple([], [], [], [], [], [], [])
        reserved_cards_mask = self.append_tuples(reserved_cards_list, [self.card_to_input(card) for card in players_hand.cards_reserved], 3, FALSE_CARD)
        victory_points = players_hand.number_of_my_points()
        list_of_tensors = []
        for x in self.gems_to_input(players_hand.gems_possessed):
            list_of_tensors.append(np.array(x).reshape(1, 1))
        for x in self.price_to_input(players_hand.discount()):
            list_of_tensors.append(np.array(x).reshape(1,1))
        for x in reserved_cards_list:
            list_of_tensors.append(np.array(x).reshape(1,3))
        list_of_tensors.append(np.array([players_hand.number_of_my_points()]).reshape(1,1))
        list_of_tensors.append(np.array([len(players_hand.nobles_possessed)]).reshape(1, 1))
        list_of_tensors.append(np.array(reserved_cards_mask).reshape(1, 3))

        return list_of_tensors

    def state_to_input(self, state: State):
        return self.board_to_input(state.board) + self.players_hand_to_input(state.active_players_hand()) + \
               self.players_hand_to_input(state.previous_players_hand())

    def many_states_to_input(self, list_of_states: List[State]):
        input_len = 62
        output_list = [[] for _ in range(input_len)]
        for state in list_of_states:
            state_transfored = self.state_to_input(state)
            for i in range(input_len):
                output_list[i].append(state_transfored[i])

        output_list_of_tensors = []
        for i in range(input_len):
            concatated = np.concatenate(output_list[i], axis=0)
            output_list_of_tensors.append(concatated)

        return output_list_of_tensors


# xxx = Vectorizer().many_states_to_input([state_3, state_3])
# print(xxx[8].shape)
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