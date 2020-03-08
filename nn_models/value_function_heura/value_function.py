import numpy as np

from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State


class ValueFunction:

    def __init__(self):
        self.weights = [1000, 800, 500, 650, 0, -2000, 0, 0, 0, 0, 0, 0, 150, 180, 10, 10, 0, 0, 0, 0]
        self.scaling_factor = 1/30000

        # val = 0.96 [1000, 800, 500, 650, 0, -2000, 0, 0, 0, 0, 0, 0, 150, 180, 10, 20, 0, 0, 0, 0]
        #val = 0.95  [1000, 800, 500, 650, 0, -2000, 0, 0, 0, 0, 0, 0, 150, 200, 10, 20, 0, 0, 0, 0]
        #val = 0.9  [1000, 800, 500, 650, 0, -2000, 0, 0, 0, 0, 0, 0, 150, 250, 10, 20, 0, 0, 0, 0]
        # val = 0.8 [1000, 700, 500, 650, 0, -2000, 0, 0, 0, 0, 0, 0, 150, 350, 10, 20, 0, 0, 0, 0]
        # val = 0.7 self.weights = [1000, 700, 500, 450, 10, -1500, 0, 0, 0, 0, 0, 0, 150, 350, 20, 20, 0, 0, 0, 0]
        #self.weights =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #make sense 1 variant:
        #[500, 5, 5, 1, 0.2, -500, -0.2, -0.2, -0.2, -0.2, 5, 10, 0.5, 0.5, -1, -0.04, -0.04, -0.2]

    def set_weights(self, weights):
        self.weights = weights

    def pre_card_frac_value(self, gems : GemsCollection, price : GemsCollection):
        s = 0
        total_price = sum(list(price.gems_dict.values()))
        for color in GemColor:
            if price.gems_dict[color] > 0:
                s += (gems.gems_dict[color] - price.gems_dict[color]) / total_price
        s += gems.gems_dict[GemColor.GOLD] / total_price
        return s


    def cards_stats(self, state, active: bool):
        p_hand = state.active_players_hand() if active else state.other_players_hand()
        discount = p_hand.discount()

        cards_affordability = sum([int(p_hand.can_afford_card(card, discount)) for card in state.board.cards_on_board]) + \
                              sum([int(p_hand.can_afford_card(card, discount)) for card in
                                   p_hand.cards_reserved])
        value_affordability = sum([card.victory_points*int(p_hand.can_afford_card(card, discount)) for card in state.board.cards_on_board]) + \
                              sum([card.victory_points * int(
                                  p_hand.can_afford_card(card, discount)) for card in
                                   p_hand.cards_reserved])
        cards_frac_value = sum([self.pre_card_frac_value(p_hand.discount() + p_hand.gems_possessed, card.price) for card in
                                state.board.cards_on_board])
        nobles_frac_value = sum([self.pre_card_frac_value(p_hand.discount(), noble.price) for noble in state.board.nobles_on_board])

        return [cards_affordability, value_affordability, cards_frac_value, nobles_frac_value]

    def hand_features(self, players_hand : PlayersHand):
        points = players_hand.number_of_my_points()
        cards_possessed =  len(players_hand.cards_possessed)
        nobles_possessed = len(players_hand.nobles_possessed)
        total_gems_non_gold = players_hand.gems_possessed.sum()
        gem_gold = players_hand.gems_possessed.gems_dict[GemColor.GOLD]
        winner = int(points >= POINTS_TO_WIN)
        return [winner, points, cards_possessed, nobles_possessed, total_gems_non_gold, gem_gold]

    def state_to_features(self, state : State):
        my_hand = self.hand_features(state.active_players_hand())
        opp_hand = self.hand_features(state.other_players_hand())
        my_cards_stats = self.cards_stats(state, True)
        opp_cards_stats = self.cards_stats(state, False)
        return my_hand + opp_hand + my_cards_stats + opp_cards_stats

    #[1000, 700, 500, 650, 0, -1000, 0, 0, 0, 0, 0, 0, 150, 350, 20, 30, 0, 0, 0, 0]


    def evaluate(self, state: State):
        #print(self.weights)
        #print(np.array(self.state_to_features(state)))
        value =  np.dot(np.array(self.weights), np.array(self.state_to_features(state)))
        return value*self.scaling_factor

# gui = SplendorGUI()
# vf = ValueFunction()
# for st in list_of_fixes_states:
#     print(vf.evaluate(st))
#print(vf.state_to_features(state_x))
#print(StateAsDict(state_x))
# gui.draw_state(state_x)
# gui.keep_window_open(100)
