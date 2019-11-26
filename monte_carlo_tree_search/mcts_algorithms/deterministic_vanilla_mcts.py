from typing import List

import math
import numpy as np

from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.mcts import MCTS
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRolloutPolicy
from monte_carlo_tree_search.score_compute import UCB1Score
from monte_carlo_tree_search.tree import TreeNode
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode


class DeterministicVanillaMCTS(MCTS):
    def __init__(self,
                 iteration_limit=None,
                 exploration_parameter = 1/math.sqrt(2),
                 rollout_policy=RandomRolloutPolicy(distribution='first_buy'),
                 rollout_repetition = 10):


        super().__init__(iteration_limit=iteration_limit,
                         rollout_policy= rollout_policy,
                         rollout_repetition=rollout_repetition,
                         environment_id='splendor-deterministic-v0')


        self.exploration_parameter = exploration_parameter
        self.rollout_policy = rollout_policy
        self.score_evaluator = UCB1Score(self.exploration_parameter)
        self.root = None

    def create_root(self, state):
        self.original_root = DeterministicTreeNode(state=state, parent=None, parent_action=None, reward=0,
                                                   is_done=False, winner_id=None)
        self.root = self.original_root

    def change_root(self, node):
        self.root = node

    def _rollout_from_state_as_dict(self, state_as_dict: StateAsDict):
        value = 0
        is_done = False
        self.env.load_state_from_dict(state_as_dict)
        winner_id = None
        while not is_done:
            action = self.rollout_policy.choose_action(self.env.current_state_of_the_game)
            if action is not None:
                new_state, reward, is_done, winner_id = self.env.deterministic_step(action)
                value = reward
            else:
                winner_id = self.env.previous_player_id()
                value = 0.1
                break
        return winner_id, value

    def _rollout(self, leaf: TreeNode):

        if not leaf.terminal:
            value = 0
            is_done = False
            self.env.load_state_from_dict(leaf.state_as_dict)
            winner_id = None
            while not is_done:
                action = self.rollout_policy.choose_action(self.env.current_state_of_the_game)
                if action is not None:
                    new_state, reward, is_done, winner_id = self.env.deterministic_step(action)
                    value = reward
                else:
                    winner_id = self.env.previous_player_id()
                    value = 0.1
                    break

            return winner_id, value

        if leaf.terminal:
            value = 0
            winner_id = leaf.winner_id
            if leaf.perfect_value is not None:
                value = leaf.perfect_value

            return winner_id, value


    def move_root(self, action):
        if self.root.expanded() == False:
            self._expand_leaf(self.root)
        root_active = self.root.state.active_player_id
        print(' _______  \n ROOT ACTIVE PLAYER {} \n ROOT STAE = {}, \n'.format(root_active, self.root.state_as_dict))
        self.root = self.root.action_to_children_dict[action.__repr__()]

    def _tree_traversal(self):
        search_path = [self.root]
        while search_path[-1].expanded() and not search_path[-1].terminal:
            node_to_add = self._select_child(search_path[-1])
            search_path.append(node_to_add)

        return search_path[-1], search_path

    def _expand_leaf(self, leaf: DeterministicTreeNode):
        if not leaf.expanded():
            leaf.generate_actions()
            leaf.check_if_terminal()
            for action in leaf.actions:
                self.env.load_state_from_dict(leaf.state_as_dict)
                child_state, reward, is_done, winner_id = self.env.deterministic_step(action)
                new_child = DeterministicTreeNode(child_state, leaf, action, reward, is_done, winner_id)
                leaf.action_to_children_dict[action.__repr__()] = new_child
                leaf.children.append(new_child)

    def _select_child(self, node):
        children_ratings = [self.score_evaluator.compute_score(child, node) for child in node.children]
        child_to_choose_index = np.argmax(children_ratings)

        return node.children[child_to_choose_index]

    def _select_best_child(self, node: DeterministicTreeNode=None):
        node = node if node is not None else self.root
        children_values = [child.value_acc.get() for child in node.children if child.value_acc.get() is not None]
        best_child_index = np.argmax(children_values)
        if len(children_values) < len(node.children):
            print('\n WARNING: MCTS has not evaluated all possible moves. Choosing from a subset. \n')
        return node.children[best_child_index], node.actions[best_child_index]


    def _backpropagate(self, search_path: List[DeterministicTreeNode], winner_id, value):
        for node in search_path:
            if winner_id is not None:
                if node.active_player_id() == winner_id:
                    node.value_acc.add(-value)
                else:
                    node.value_acc.add(value)
            else:
                node.value_acc.add(0)

    def choose_action(self):
        _, best_action = self._select_best_child()
        return best_action



