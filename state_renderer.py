from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

dict = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {1, 3, 36, 4, 39, 18, 21, 23, 59}, 'cards_reserved_ids': {82, 45, 30}, 'gems_possessed': [0, 0, 0, 1, 0, 2], 'name': 'RandomAgent - first_buy '}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {40, 42, 76, 15, 54, 56, 57, 58}, 'cards_reserved_ids': {72, 33, 44}, 'gems_possessed': [0, 5, 2, 3, 5, 0], 'name': 'MCTS'}, 'board': {'nobles_on_board': {104, 105, 101}, 'cards_on_board': {34, 6, 71, 8, 73, 11, 78, 47, 48, 17, 20, 89}, 'gems_on_board': [5, -1, 2, 0, -1, 2], 'deck_order': [{'Row.CHEAP': [38, 19, 5, 37, 7, 41, 43, 22, 79, 60, 74, 25, 24, 0, 75, 77, 2, 61, 55]}, {'Row.MEDIUM': [13, 67, 49, 85, 31, 27, 10, 28, 26, 81, 66, 62, 12, 65, 63, 84, 29, 83, 80, 46, 64, 9]}, {'Row.EXPENSIVE': [68, 14, 16, 32, 87, 52, 50, 88, 53, 51, 86, 69, 35, 70]}]}, 'active_player_id': 1}

stanek = StateAsDict()
stanek.load_from_dict(dict)

fufu = SplendorGUI()
fufu.draw_state(stanek.to_state())
fufu.keep_window_open(300)