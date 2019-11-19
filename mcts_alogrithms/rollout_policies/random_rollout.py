import random

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from mcts_alogrithms.rolluot_policy import RolloutPolicy
from gym_splendor_code.envs.mechanics.action_space_generator_fast import generate_all_legal_reservations
from gym_splendor_code.envs.mechanics.action_space_generator_fast import generate_all_legal_buys
from gym_splendor_code.envs.mechanics.action_space_generator_fast import generate_all_legal_trades

class RandomRolloutPolicy(RolloutPolicy):

    def __init__(self, distribution: str = 'unforim'):
        super().__init__('random')
        self.distribution = distribution
    def choose_action(self, state : State) ->Action:
        # first we load observation to the private environment
        actions_by_type = {'buy' : generate_all_legal_buys(state), 'trade' : generate_all_legal_trades(state),
                       'reserve': generate_all_legal_reservations(state)}

        list_of_actions = actions_by_type['buy'] + actions_by_type['reserve'] + actions_by_type['trade']

        if len(list_of_actions):
            if self.distribution == 'uniform':
                return random.choice(list_of_actions)
            if self.distribution == 'uniform_on_types':
                chosen_action_type = random.choice([action_type for action_type in
                                                    actions_by_type.keys() if
                                                    len(actions_by_type[action_type]) > 0])
                return random.choice(actions_by_type[chosen_action_type])
            if self.distribution == 'first_buy':
                if len(actions_by_type['buy']) > 0:
                    return random.choice(actions_by_type['buy'])
                else:
                    return random.choice(list_of_actions)

        else:
            return None
