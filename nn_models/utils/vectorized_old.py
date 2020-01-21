import numpy as np

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

    def one_hot(self, i, n):
        vec = np.zeros(n)
        vec[i] = 1
        return vec

    def price_to_tuple(self, gems_collection : GemsCollection):
       return PriceTuple(*[gems_collection.gems_dict[gem_color] for gem_color in GemColor if gem_color !=
             GemColor.GOLD])

    def gems_collection_to_tuple(self, gems_collection : GemsCollection, n=6):
        return GemsTuple(*[gems_collection.gems_dict[gem_color] for gem_color in GemColor])

    def card_to_tuple(self, card):
        profit_vec = card.discount_profit.value-1
        price_vec = self.price_to_tuple(card.price)
        victory_points_vec = card.victory_points
        return CardTuple(profit_vec, price_vec, victory_points_vec)

    def noble_to_tuple(self, noble):
        return self.price_to_tuple(noble.price)

    def players_hand_to_vectors(self, players_hand: PlayersHand):
        discount = self.price_to_tuple(players_hand.discount())
        gems_possessed = self.gems_collection_to_tuple(players_hand.gems_possessed)
        nobles_posessed = self.one_hot(len(players_hand.nobles_possessed), 4)
        cards_reserved_list = []
        for card in players_hand.cards_reserved:
            cards_reserved_list.append(self.card_to_tuple(card))
        cards_reserved_list = cards_reserved_list + [None]*(3 - len(cards_reserved_list))
        victory_points = players_hand.number_of_my_points()
        return PlayerTuple(discount, gems_possessed, cards_reserved_list, victory_points, nobles_posessed)

    def board_to_vectors(self, board: Board):
        gems_on_board = self.gems_collection_to_tuple(board.gems_on_board)
        cards_on_board_list = []
        for card in board.cards_on_board:
            cards_on_board_list.append(self.card_to_tuple(card))
        nobles_on_board_list = []
        for noble in board.nobles_on_board:
            nobles_on_board_list.append(self.noble_to_tuple(noble))

        return BoardTuple(gems_on_board, cards_on_board_list, nobles_on_board_list)

    def observation_to_vector(self, observation : DeterministicObservation):

        local_state = observation.recreate_state()
        active_player = self.players_hand_to_vectors(local_state.active_players_hand())
        previous_player = self.players_hand_to_vectors(local_state.previous_players_hand())
        board_info = self.board_to_vectors(local_state.board)

        return ObservationTuple(active_player, previous_player, board_info)

