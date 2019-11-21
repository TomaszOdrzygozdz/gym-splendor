from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

dict = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {3, 41, 42, 78, 79, 58, 59, 61}, 'cards_reserved_ids': {32, 43, 53}, 'gems_possessed': [0, 1, 4, 0, 1, 0], 'name': 'MCTS'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {73, 20}, 'cards_reserved_ids': {64, 21, 46}, 'gems_possessed': [0, 3, 0, 1, 3, 3], 'name': 'RandomAgent - uniform '}, 'board': {'nobles_on_board': {105, 102, 103}, 'cards_on_board': {34, 38, 8, 9, 76, 16, 84, 52, 87, 56, 57, 30}, 'gems_on_board': [5, 0, 0, 3, 0, 1], 'deck_order': [{'Row.CHEAP': [1, 22, 7, 4, 5, 19, 75, 72, 39, 2, 77, 6, 37, 74, 60, 18, 23, 36, 55, 24, 0, 40, 54, 25]}, {'Row.MEDIUM': [63, 12, 67, 13, 62, 44, 47, 48, 31, 49, 26, 11, 81, 66, 45, 27, 83, 85, 10, 65, 28, 29, 82, 80]}, {'Row.EXPENSIVE': [50, 15, 51, 71, 33, 17, 89, 86, 88, 14, 69, 68, 70, 35]}]}, 'active_player_id': 0}

Shower = SplendorGUI()
sad = StateAsDict(None)
sad.load_from_dict(dict)
stanek = sad.to_state()

Shower.draw_state(stanek)
Shower.keep_window_open(300)