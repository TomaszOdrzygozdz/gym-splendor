from typing import List

import math
import numpy as np

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.mcts_algorithms.abstract_deterministic_mcts import MCTS
from monte_carlo_tree_search.rollout_policies.greedy_rollout import GreedyRolloutPolicy
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRolloutPolicy

from monte_carlo_tree_search.score_computers.ucb1_score import UCB1Score
from monte_carlo_tree_search.trees.abstract_tree import TreeNode
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode


class DeterministicMCTSVanillaRollout(MCTS):
    def __init__(self,
                 iteration_limit=None,
                 exploration_parameter = 1/math.sqrt(2),
                 rollout_policy = "random",
                 type = None,
                 rollout_repetition = 10):

        if rollout_policy == "random":
            if type is None:
                type = "first_buy"
            rollout = RandomRolloutPolicy(distribution = type)

        elif rollout_policy == "greedy":
            if type is None:
                type = [100, 2, 2, 1, 0.1]
            rollout = GreedyRolloutPolicy(weight = type)


        super().__init__(iteration_limit=iteration_limit,
                         rollout_policy= rollout,
                         rollout_repetition=rollout_repetition,
                         environment_id='splendor-v0')


        self.exploration_parameter = exploration_parameter
        self.score_evaluator = UCB1Score(self.exploration_parameter)
        self.root = None

    def create_root(self, observation: DeterministicObservation):
        print(observation)
        self.original_root = DeterministicTreeNode(observation=observation, parent=None, parent_action=None, reward=0,
                                                   is_done=False, winner_id=None)
        self.root = self.original_root

    def change_root(self, node):
        self.root = node

    def _rollout(self, observation: DeterministicObservation):
        value = 0
        is_done = False
        self.env.load_observation(observation)
        winner_id = None
        while not is_done:
            action = self.rollout_policy.choose_action(self.env.current_state_of_the_game)
            if action is not None:
                _, reward, is_done, info = self.env.step('deterministic',action, return_observation=False)
                winner_id = info['winner_id']
                value = reward
            else:
                winner_id = self.env.previous_player_id()
                value = 0.1
                break
        return winner_id, value

    def move_root(self, action):
        if self.root.expanded() == False:
            self._expand_leaf(self.root)
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
                self.env.load_observation(leaf.observation)
                child_state_observation, reward, is_done, info = self.env.step('deterministic', action)
                winner_id = info['winner_id']
                new_child = DeterministicTreeNode(child_state_observation, leaf, action, reward, is_done, winner_id)
                leaf.action_to_children_dict[action.__repr__()] = new_child
                leaf.children.append(new_child)

    def _select_child(self, node):
        children_ratings = [self.score_evaluator.compute_score(child, node) for child in node.children]
        child_to_choose_index = np.argmax(children_ratings)
        return node.children[child_to_choose_index]

    def _select_best_child(self, node: DeterministicTreeNode=None):
        node = node if node is not None else self.root
        children_values = [child.value_acc.get() for child in node.children if child.value_acc.get() is not None]
        if len(children_values):
            best_child_index = np.argmax(children_values)
            if len(children_values) < len(node.children):
                print('\n WARNING: MCTS has not evaluated all possible moves. Choosing from a subset. \n')
            return node.children[best_child_index], node.actions[best_child_index]
        else:
            print(print('\n WARNING: MCTS has not evaluated all possible moves. Choosing from a subset. \n'))
            return None, None


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
