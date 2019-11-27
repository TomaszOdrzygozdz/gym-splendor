import random

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_reservations
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_buys
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_trades
from monte_carlo_tree_search.rolluot_policy import RolloutPolicy


class RandomRolloutPolicy(RolloutPolicy):

    def __init__(self,
                    weight: list = [100,2,2,1.0.1]):
        super().__init__('greedy')
        self.weight = weight

    def choose_action(self, observation, previous_actions) -> Action:
        actions_by_type = {'buy' : generate_all_legal_buys(state), 'trade' : generate_all_legal_trades(state),
                       'reserve': generate_all_legal_reservations(state)}

        list_of_actions = actions_by_type['buy'] + actions_by_type['reserve'] + actions_by_type['trade']
        current_points = state.active_players_hand().number_of_my_points()

        if len(list_of_actions):
            actions = []
            points = []
            for action in list_of_actions:
                ae = action.evaluate(state)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                    self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                actions.append(action)
                points.append(potential_reward)
            actions = [actions[i] for i, point in enumerate(points) if  point >= sorted(set(points))[-1]]
            return random.choice(actions)

        else:
            return None
