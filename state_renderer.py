from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

dict = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': set(), 'gems_possessed': [0, 0, 1, 1, 0, 1], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': {28, 20}, 'gems_possessed': [2, 0, 0, 0, 0, 0], 'name': 'RandomAgent - uniform_on_types '}, 'board': {'nobles_on_board': {104, 108, 102}, 'cards_on_board': {33, 4, 69, 71, 72, 41, 9, 10, 44, 79, 86, 26}, 'gems_on_board': [2, 4, 3, 3, 4, 3], 'deck_order': [{'Row.CHEAP': [38, 0, 76, 18, 55, 19, 3, 40, 7, 43, 23, 39, 37, 42, 77, 75, 25, 59, 2, 5, 54, 73, 56, 21, 57, 22, 1, 36, 6, 60, 61, 78, 74, 58, 24]}, {'Row.MEDIUM': [12, 49, 47, 31, 45, 62, 11, 81, 46, 80, 66, 67, 48, 83, 82, 65, 8, 84, 13, 29, 27, 64, 85, 63, 30]}, {'Row.EXPENSIVE': [35, 53, 87, 88, 15, 14, 50, 17, 16, 68, 70, 32, 52, 51, 89, 34]}]}, 'active_player_id': 1}


stanek = StateAsDict()
stanek.load_from_dict(dict)

fufu = SplendorGUI()
fufu.draw_state(stanek.to_state())
fufu.keep_window_open(300)