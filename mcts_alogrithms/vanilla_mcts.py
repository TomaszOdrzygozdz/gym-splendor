from typing import List, Tuple

import time

import gym
import math
import numpy as np
import random

from gym_splendor_code.envs.mechanics.state import State
from mcts_alogrithms.rollout_policies.random_rollout import RandomRolloutPolicy
from mcts_alogrithms.tree import TreeNode


class FullStateVanillaMCTS:
    def __init__(self,
                 time_limit=None,
                 iteration_limit=None,
                 exploration_parameter = 0.3,
                 rollout_policy=RandomRolloutPolicy(),
                 rollout_iteration_limit=30):

        self.full_state_env = gym.make('splendor-full_state-v0')

        if time_limit != None:
            if iteration_limit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = time_limit
            self.limitType = 'time'
        else:
            if iteration_limit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iteration_limit
            self.limitType = 'iterations'
        self.exploration_parameter = exploration_parameter
        self.rollout_policy = rollout_policy

    def create_root(self, state):
        self.root = TreeNode(state=state, parent=None, terminal=False, winner_id=None)


    def run_mcts_pass(self, root: TreeNode):

        leaf, search_path = self._tree_traversal(root)
        self._expand_leaf(leaf)
        node_to_rollout = self._select_child(leaf)
        winner_id = self._rollout(leaf)
        self._backpropagate(search_path, winner_id)


    def _tree_traversal(self, root):
        node = root
        search_path = []
        while node.expanded():
            new_node, action = self._select_child(node)
            search_path.append((node, action))
            node = new_node
            if new_node is None:
                break
        return node, search_path

    def _expand_leaf(self, leaf: TreeNode):
        # First generate all legal actions in a leaf:
        leaf.generate_actions()
        for action in leaf.actions:
            self.full_state_env.load_state_from_dict(leaf.state_as_dict)
            child_state, reward, is_done, who_won = self.full_state_env.full_state_step(action)
            new_child = TreeNode(child_state, leaf, is_done)
            leaf.children.append(new_child)

    def _rollout(self, leaf: TreeNode):
        self.full_state_env.load_state_from_dict(leaf.state_as_dict)
        is_done = leaf.terminal
        winner_id = None
        while is_done is False:
            action = self.rollout_policy.choose_action(self.full_state_env.current_state_of_the_game)
            if action is None:
                break
            else:
                new_state, reward, is_done, winner_id = self.full_state_env.full_state_step(action)
            if is_done:
                winner_id = self.full_state_env.active_player_id

    def _select_child(self, node):
        p = np.random.uniform(0,1)
        if p < self.exploration_parameter:
            return self._select_random_child(node)
        else:
            return self._select_child(node)

    def _select_best_child(self, node):
        children_values = [child.value_acc.get_value for child in node.children]
        best_child_index = np.argmax(children_values)
        return node.children[best_child_index], node.actions[best_child_index]

    def _select_random_child(self, node):
        return random.choice(node.children)

    def _backpropagate(self, search_path: List[Tuple], winner_id):
        value = 1
        for node_action_pair in search_path:
            node, action = node_action_pair
            if winner_id is not None:
                if node.active_player_id == winner_id:
                    node.value_acc.add(value)
                else:
                    node.value_acc.add(-value)
            else:
                node.value_acc.add(0)

    def run_simulation(self):
        assert self.root is not None, 'Root is None. Cannot run MCTS pass.'
        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.run_mcts_pass(self.root)
        else:
            for i in range(self.searchLimit):
                self.run_mcts_pass(self.root)

    def choose_best_action(self, root=None):
        if root is None:
            root = self.root
        children_values = [child.value_acc.get() for child in root.children]
        best_child_index = np.argmax(children_values)
        return root.actions[best_child_index]












