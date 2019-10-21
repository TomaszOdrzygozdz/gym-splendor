# w1 = GemsCollecion({GemColor.GOLD: 1, GemColor.RED: 2, GemColor.GREEN: 5, GemColor.BLUE: 3, GemColor.WHITE: 4, GemColor.BLACK: 5})
# w2 = GemsCollecion({GemColor.GOLD: 2, GemColor.RED: 1, GemColor.GREEN: 6, GemColor.BLUE: 3, GemColor.WHITE: 7, GemColor.BLACK: 5})
# print(w1 <= w2)
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI, GemColor
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_reservations
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State

s = State()
f = SplendorGUI()

f.draw_board(s.board, 200, 10)
f.keep_window_open()

#testing action generator
pla = PlayersHand()
pla.gems_possessed.gems_dict[GemColor.BLUE] = 4
pla.gems_possessed.gems_dict[GemColor.GREEN] = 4
pla.gems_possessed.gems_dict[GemColor.RED] = 2
pla.gems_possessed.gems_dict[GemColor.WHITE] = 0
pla.gems_possessed.gems_dict[GemColor.BLACK] = 0
pla.gems_possessed.gems_dict[GemColor.GOLD] = 0
d = State()
d.list_of_players_hands = [pla, PlayersHand()]
print(d.active_players_hand().gems_possessed)
f = generate_all_legal_reservations(d)
print(len(f))
for du  in f:
    print(du)