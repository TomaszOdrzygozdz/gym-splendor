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

class Vectorizer:

    def __init__(self):
        pass

    def one_hot(self, i, n):
        vec = np.zeros(n)
        vec[i] = 1
        return vec

    def gems_collection_to_vector(self, gems_collection : GemsCollection, with_gold = True, n=5):
        if not with_gold:
            list_of_vectors = [self.one_hot(gems_collection.gems_dict[gem_color]-1, n+1) for gem_color in GemColor if gem_color !=
             GemColor.GOLD]
        else:
            list_of_vectors = [self.one_hot(gems_collection.gems_dict[gem_color], n+1) for gem_color in GemColor]
        return np.concatenate(list_of_vectors)

    def card_to_vector(self, card):
        profit_vec = self.one_hot(card.discount_profit.value-1, 5)
        price_vec = self.gems_collection_to_vector(card.price, with_gold=False, n=7)
        victory_points_vec = self.one_hot(card.victory_points, 6)
        return np.concatenate([profit_vec, price_vec, victory_points_vec])

    def noble_to_vector(self, noble):
        price_vec = self.gems_collection_to_vector(noble.price, with_gold=False, n=4)
        return price_vec

    def players_hand_to_vectors(self, players_hand: PlayersHand):
        discount_vec = self.gems_collection_to_vector(players_hand.discount(), with_gold=False, n=15)
        gems_possessed_vec = self.gems_collection_to_vector(players_hand.gems_possessed, with_gold=True, n=5)
        nobles_posessed_vec = self.one_hot(len(players_hand.nobles_possessed), 4)
        cards_reserved_list = []
        for card in players_hand.cards_reserved:
            cards_reserved_list.append(self.card_to_vector(card))
        cards_reserved_list = cards_reserved_list + [None]*(3 - len(cards_reserved_list))
        victory_points_vec = self.one_hot(players_hand.number_of_my_points(), 25)

        return {'discount_vec': discount_vec, 'gems_vec' : gems_possessed_vec,
                'cards_reserved_vec' : cards_reserved_list, 'points_vec': victory_points_vec,
                'nobles_vec' : nobles_posessed_vec}

    def board_to_vectors(self, board: Board):
        gems_on_board_vec = self.gems_collection_to_vector(board.gems_on_board, with_gold=True, n=5)
        cards_on_board_list = []
        for card in board.cards_on_board:
            cards_on_board_list.append(self.card_to_vector(card))
        nobles_on_board_list = []
        for noble in board.nobles_on_board:
            nobles_on_board_list.append(self.noble_to_vector(noble))

        return {'gems_on_board_vec' : gems_on_board_vec, 'cards_on_board_list' : cards_on_board_list,
                'nobles_on_board_list' : nobles_on_board_list}


    def observation_to_vector(self, observation : DeterministicObservation):

        local_state = observation.recreate_state()

        active_players_hand_vec = self.players_hand_to_vectors(local_state.active_players_hand())
        previous_players_hand_vec = self.players_hand_to_vectors(local_state.previous_players_hand())
        board_vectors = self.board_to_vectors(local_state.board)

        return {'active_players_hand_vec' : active_players_hand_vec,
                'previous_players_hand_vec' : previous_players_hand_vec,
                'board_vectors' : board_vectors}


fufer = Vectorizer()
x = fufer.observation_to_vector(DeterministicObservation(state_3))
print(len(x['active_players_hand_vec']['nobles_vec']))

# xoxo = SplendorGUI()
# xoxo.draw_state(state_3)