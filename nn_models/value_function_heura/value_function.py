from archive.states_list import list_of_fixes_states
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI, GemColor
from gym_splendor_code.envs.mechanics.board import Board
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


class ValueFunction:

    def __init__(self):
        pass

    def card_frac_value(self, gems : GemsCollection, price : GemsCollection):
        s = 0
        total_price = sum(list(price.gems_dict.values()))
        for color in GemColor:
            if price.gems_dict[color] > 0:
                s += (gems.gems_dict[color] - price.gems_dict[color]) / total_price
        s += gems.gems_dict[GemColor.GOLD] / total_price
        return s

    def cards_stats(self, state):
        discount = state.active_players_hand().discount()
        cards_affordability = sum([int(state.active_players_hand().can_afford_card(card, discount)) for card in state.board.cards_on_board])
        value_affordability = sum([card.victory_points*int(state.active_players_hand().can_afford_card(card, discount)) for card in state.board.cards_on_board])
                               # [reserved_card for reserved_card in state.active_players_hand().cards_reserved if
                               #  state.active_players_hand().can_afford_card(reserved_card, discount)]
        cards_frac_value = sum([self.card_frac_value(state.active_players_hand().discount() + state.active_players_hand().gems_possessed, card.price) for card in
                           state.board.cards_on_board])
        return [cards_affordability, value_affordability, cards_frac_value]

    def hand_features(self, players_hand : PlayersHand, board : Board):
        ft = []
        ft.append(players_hand.number_of_my_points())
        ft.append(len(players_hand.cards_possessed))

        return ft

    def state_to_features(self, state : State):
        ft1 = self.hand_features(state.active_players_hand(), state.board)
        ft2 = self.hand_features(state.other_players_hand(), state.board)
        ft3 = self.cards_stats(state)
        return ft3

gui = SplendorGUI()
state_x = list_of_fixes_states[6]
vf = ValueFunction()
print(vf.state_to_features(state_x))
print(StateAsDict(state_x))
# gui.draw_state(state_x)
# gui.keep_window_open(100)