from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import CARDS_IN_DECK, NOBLES_IN_DECK, GemColor
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
import numpy as np

def vectorize_state(state_as_dict: StateAsDict):

    state = state_as_dict.state_as_dict
    output = []
    gem_list = []
    for i in state.keys():
        if isinstance(state[i], dict):
            for j in state[i].keys():
                if "card" in j:
                    output.extend([1 if y in state[i][j] else 0 for y in np.arange(CARDS_IN_DECK)])
                elif "noble" in j:
                    output.extend([1 if y in state[i][j] else 0 for y in np.arange(NOBLES_IN_DECK)])
                elif "gems" in j:
                    gem_list.extend(state[i][j])
    output.extend(gem_list)
    return output

def vectorize_action(pure_action : Action):

    action_as_dict = pure_action.to_dict()

    if action_as_dict["action_type"] == "buy":
        action = [1,0,0]
    elif action_as_dict["action_type"] == "reserve":
        action = [0,1,0]
    else:
        action = [0,0,1]

    action.extend([1 if y in {action_as_dict["card"]} else 0 for y in np.arange(CARDS_IN_DECK)])
    action.extend(action_as_dict["gems_flow"])
    return action