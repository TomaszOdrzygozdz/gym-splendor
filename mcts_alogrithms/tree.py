import random
from typing import Dict
# tree node
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from mcts_alogrithms.value_accumulators import ScalarMeanMaxValueAccumulator


class TreeNode:
    id = 0

    def __init__(self, state: State, parent: 'MCTSTreeNode', parent_action: Action, terminal: bool = False)->None:
        self.id = TreeNode.id
        TreeNode.id += 1
        self.parent = parent
        self.parent_action = parent_action
        self.state_as_dict = StateAsDict(state)
        self.state = self.state_as_dict.to_state()
        self.actions = []
        self.action_to_children_dict = {}
        self.children = []
        self.value_acc = ScalarMeanMaxValueAccumulator()
        if parent is None:
            self.generation = 0
        else:
            self.generation = parent.generation + 1

    def get_id(self):
        return self.id

    def active_player_id(self):
        return self.state.active_player_id

    def check_if_terminal(self):
        self.terminal = True if len(self.actions) == 0 else False

    def generate_actions(self):
        if len(self.actions) == 0:
            self.actions = generate_all_legal_actions(self.state)

    def state(self):
        return self.state

    def expanded(self):
        return True if self.actions else False

    def terminal(self):
        return self.terminal

    def parent(self):
        return self.parent

    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False


