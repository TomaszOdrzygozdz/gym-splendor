from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

d1 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': {2}, 'gems_possessed': [1, 1, 1, 1, 0, 0], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': set(), 'gems_possessed': [0, 2, 1, 0, 1, 2], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {108, 102, 103}, 'cards_on_board': {69, 44, 14, 78, 79, 82, 51, 85, 53, 55, 58, 62}, 'gems_on_board': [4, 1, 2, 3, 3, 2], 'deck_order': [{'Row.CHEAP': [6, 56, 20, 73, 59, 25, 43, 72, 3, 76, 21, 39, 0, 36, 38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [49, 27, 9, 31, 80, 8, 28, 10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [87, 33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}


d1_1 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': {2}, 'gems_possessed': [1, 2, 2, 1, 0, 1], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': set(), 'gems_possessed': [0, 2, 2, 1, 1, 3], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {108, 102, 103}, 'cards_on_board': {69, 44, 14, 78, 79, 82, 51, 85, 53, 55, 58, 62}, 'gems_on_board': [4, 0, 0, 2, 3, 0], 'deck_order': [{'Row.CHEAP': [6, 56, 20, 73, 59, 25, 43, 72, 3, 76, 21, 39, 0, 36, 38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [49, 27, 9, 31, 80, 8, 28, 10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [87, 33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}

d1_2 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': set(), 'cards_reserved_ids': {2, 14}, 'gems_possessed': [2, 2, 2, 1, 0, 1], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {55}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 1, 2, 1, 0, 0], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {108, 102, 103}, 'cards_on_board': {69, 6, 44, 78, 79, 82, 51, 85, 53, 87, 58, 62}, 'gems_on_board': [3, 1, 0, 2, 4, 3], 'deck_order': [{'Row.CHEAP': [56, 20, 73, 59, 25, 43, 72, 3, 76, 21, 39, 0, 36, 38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [49, 27, 9, 31, 80, 8, 28, 10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}

d2 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {2, 78, 79, 58, 59}, 'cards_reserved_ids': {49, 82, 14}, 'gems_possessed': [1, 3, 0, 1, 1, 2], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {3, 6, 73, 44, 20, 55, 56, 25, 62}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 1, 0, 3, 2, 2], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {108, 102, 103}, 'cards_on_board': {69, 72, 9, 43, 76, 51, 21, 85, 53, 87, 27, 31}, 'gems_on_board': [4, 0, 4, 0, 1, 0], 'deck_order': [{'Row.CHEAP': [39, 0, 36, 38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [80, 8, 28, 10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}

d3 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {2, 78, 79, 58, 59}, 'cards_reserved_ids': {49, 82, 14}, 'gems_possessed': [0, 1, 2, 0, 3, 4], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': {108}, 'cards_possessed_ids': {3, 6, 39, 73, 9, 43, 44, 80, 20, 55, 56, 25, 27, 62}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 1, 0, 3, 0, 0], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {102, 103}, 'cards_on_board': {0, 69, 8, 72, 76, 51, 21, 85, 53, 87, 28, 31}, 'gems_on_board': [5, 2, 2, 1, 1, 0], 'deck_order': [{'Row.CHEAP': [36, 38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}

d3_5 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {2, 78, 79, 58, 59}, 'cards_reserved_ids': {49, 82, 14}, 'gems_possessed': [0, 1, 2, 0, 3, 4], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': {108}, 'cards_possessed_ids': {3, 6, 39, 73, 9, 43, 44, 80, 20, 55, 56, 25, 27, 62}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 1, 0, 3, 0, 0], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {102, 103}, 'cards_on_board': {0, 69, 8, 72, 76, 51, 21, 85, 53, 87, 28, 31}, 'gems_on_board': [5, 2, 2, 1, 1, 0], 'deck_order': [{'Row.CHEAP': [36, 38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}

d4 = {'active_player_hand': {'noble_possessed_ids': set(), 'cards_possessed_ids': {2, 78, 79, 58, 59}, 'cards_reserved_ids': {49, 82, 14}, 'gems_possessed': [0, 1, 3, 0, 3, 3], 'name': 'Multi Process MCTS'}, 'other_player_hand': {'noble_possessed_ids': {108, 103}, 'cards_possessed_ids': {3, 6, 39, 73, 9, 43, 44, 76, 80, 20, 55, 56, 25, 27, 62}, 'cards_reserved_ids': set(), 'gems_possessed': [0, 1, 0, 3, 0, 0], 'name': 'Greedy [100, 2, 2, 1, 0.1] 3'}, 'board': {'nobles_on_board': {102}, 'cards_on_board': {0, 36, 69, 8, 72, 51, 21, 85, 53, 87, 28, 31}, 'gems_on_board': [5, 2, 1, 1, 1, 1], 'deck_order': [{'Row.CHEAP': [38, 7, 18, 22, 4, 24, 60, 61, 23, 37, 57, 5, 41, 74, 54, 40, 77, 19, 75, 42, 1]}, {'Row.MEDIUM': [10, 67, 45, 65, 12, 29, 47, 46, 83, 26, 48, 66, 13, 30, 64, 84, 11, 81, 63]}, {'Row.EXPENSIVE': [33, 52, 34, 16, 88, 71, 15, 17, 68, 35, 50, 89, 86, 70, 32]}]}, 'active_player_id': 0}


sd1 = StateAsDict()
sd1.load_from_dict(d1)
state_1 = sd1.to_state()
state_1.active_player_id = 0

sd1_1 = StateAsDict()
sd1_1.load_from_dict(d1_1)
state1_1 = sd1_1.to_state()
state1_1.active_player_id = 1

sd1_2 = StateAsDict()
sd1_2.load_from_dict(d1_2)
state_1_2 = sd1_2.to_state()


sd2 = StateAsDict()
sd2.load_from_dict(d2)
state_2 = sd2.to_state()

sd3_5 = StateAsDict()
sd3_5.load_from_dict(d3)
state_3_5 = sd3_5.to_state()

sd3 = StateAsDict()
sd3.load_from_dict(d3)
state_3 = sd3.to_state()

sd4 = StateAsDict()
sd4.load_from_dict(d4)
state_4 = sd4.to_state()

obs4 = DeterministicObservation(state_4)

obs3 = DeterministicObservation(state_3)
obs2 = DeterministicObservation(state_2)
obs1 = DeterministicObservation(state_1)
obs1_1 = DeterministicObservation(state1_1)
obs1_2 = DeterministicObservation(state_1_2)
