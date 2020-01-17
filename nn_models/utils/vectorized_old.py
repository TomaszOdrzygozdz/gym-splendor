import numpy as np

from archive.states_list import state_3
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State

from collections import namedtuple

GemsCollectionTuple = namedtuple('gems_collection', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))
PriceTuple = namedtuple('price', ' '.join([str(x).replace('GemColor.', '') for x in GemColor if x != GemColor.GOLD]))
CardTuple = namedtuple('card', 'profit price victory_points')

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
        return GemsCollectionTuple(*[(gems_collection.gems_dict[gem_color], n) for gem_color in GemColor])

    def card_to_tuple(self, card):
        profit_vec = self.one_hot(card.discount_profit.value-1, 5)
        price_vec = self.price_to_tuple(card.price)
        victory_points_vec = self.one_hot(card.victory_points, 6)
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
        victory_points_vec = self.one_hot(players_hand.number_of_my_points(), 25)

        return {'discount_vec': discount, 'gems' : gems_possessed,
                'cards_reserved' : cards_reserved_list, 'points_vec': victory_points_vec,
                'nobles' : nobles_posessed_vec}

    def board_to_vectors(self, board: Board):
        gems_on_board_vec = self.gems_collection_to_tuple(board.gems_on_board)
        cards_on_board_list = []
        for card in board.cards_on_board:
            cards_on_board_list.append(self.card_to_tuple(card))
        nobles_on_board_list = []
        for noble in board.nobles_on_board:
            nobles_on_board_list.append(self.noble_to_tuple(noble))

        return {'gems_on_board' : gems_on_board_vec, 'cards_on_board' : cards_on_board_list,
                'nobles_on_board' : nobles_on_board_list}


    def observation_to_vector(self, observation : DeterministicObservation):

        local_state = observation.recreate_state()
        active_players_hand_vec = self.players_hand_to_vectors(local_state.active_players_hand())
        previous_players_hand_vec = self.players_hand_to_vectors(local_state.previous_players_hand())
        board_vectors = self.board_to_vectors(local_state.board)

        return {'active_player' : active_players_hand_vec,
                'previous_player' : previous_players_hand_vec,
                'board' : board_vectors}

#
# carr = state_3.board.nobles_on_board.pop()
#
fufer = Vectorizer()
# print(carr)
# x = fufer.noble_to_tuple(carr)
# print(x)

x = fufer.observation_to_vector(DeterministicObservation(state_3))
print(x)
#print(len(x['active_players_hand_vec']['nobles_vec']))

# xoxo = SplendorGUI()
# xoxo.draw_state(state_3)