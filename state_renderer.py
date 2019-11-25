from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

dict = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': {63}, 'gems_possessed': [1, 1, 0, 1, 1, 0], 'name': 'RandomAgent - uniform_on_types '}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': {72, 9}, 'gems_possessed': [2, 0, 0, 0, 0, 0], 'name': 'Multi Process MCTS'}, 'board': {'nobles_on_board': {104, 107, 109}, 'cards_on_board': {68, 71, 8, 43, 46, 48, 18, 51, 88, 25, 58, 28}, 'gems_on_board': [0, 3, 4, 3, 3, 4], 'deck_order': [{'Row.CHEAP': [19, 42, 41, 73, 6, 38, 20, 3, 4, 21, 40, 54, 56, 60, 39, 57, 76, 78, 0, 23, 2, 55, 22, 5, 24, 37, 7, 59, 74, 77, 1, 79, 75, 36, 61]}, {'Row.MEDIUM': [67, 65, 29, 64, 83, 82, 26, 45, 80, 47, 10, 62, 44, 27, 11, 31, 85, 13, 84, 12, 66, 30, 81, 49]}, {'Row.EXPENSIVE': [69, 17, 35, 50, 32, 52, 89, 16, 86, 33, 53, 70, 14, 15, 34, 87]}]}, 'active_player_id': 0}


stanek = StateAsDict()
stanek.load_from_dict(dict)

fufu = SplendorGUI()
fufu.draw_state(stanek.to_state())
fufu.keep_window_open(300)