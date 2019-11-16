from typing import Dict

# tree node
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator_fast import generate_all_legal_actions
from mcts_alogrithms.value_accumulators import ScalarMeanMaxValueAccumulator


class TreeNode(object):
    def __init__(self, state_as_dict : StateAsDict, terminal : bool = False):
        self.state_as_dict = self.state_as_dict
        self.state = State(load_from_state_as_dict=self.state_as_dict)
        self.terminal = terminal
        self.actions = generate_all_legal_actions(self.state)
        self.rewards = {action: None for action in self.actions}
        self.value_acc = ScalarMeanMaxValueAccumulator()

    def check_if_teminal(self):
        self.terminal = True if len(self.actions) > 0 else False

    @property
    def all_rewards(self):
        return self.rewards

    def reward(self, action):
        return self.rewards[action]

    @property
    def value_acc(self):
        return self.value_acc

    @property
    def state(self):
        return self.state

    def state_as_dict(self):
        return self.state_as_dict()

    def expanded(self):
        return True if self.children else False

    @property
    def terminal(self):
        return self.terminal

    @property
    def solved(self):
        return self.solved

    @terminal.setter
    def terminal(self, terminal):
        self.terminal = terminal

    @value_acc.setter
    def value_acc(self, value):
        self._value_acc = value

    @state.setter
    def state(self, value):
        self._state = value

