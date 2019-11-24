import random
from typing import Dict
# tree node
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from monte_carlo_tree_search.tree import TreeNode
from monte_carlo_tree_search.value_accumulators import ScalarMeanMaxValueAccumulator


class DeterministicTreeNode(TreeNode):

    def __init__(self, state: State, parent: 'MCTSTreeNode', parent_action: Action, terminal: bool = False)->None:
        super().__init__(parent, parent_action, ScalarMeanMaxValueAccumulator(), )
        self.state_as_dict = StateAsDict(state)
        self.state = self.state_as_dict.to_state()

    def get_id(self):
        return self.id

    def active_player_id(self):
        return self.state.active_player_id

    def check_if_terminal(self):
        self.terminal = True if len(self.actions) == 0 else False

    def generate_actions(self):
        if len(self.actions) == 0:
            self.actions = generate_all_legal_actions(self.state)
        self.check_if_terminal()

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


