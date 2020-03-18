from typing import List

import math
import numpy as np

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.mcts_algorithms.abstract_mcts import MCTS
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from monte_carlo_tree_search.score_computers.ucb1_score import UCB1Score
from monte_carlo_tree_search.trees.abstract_tree import TreeNode
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode


class SingleMCTS(MCTS):
    def __init__(self,
                 iteration_limit=None,
                 exploration_parameter = 0.4,
                 evaluation_policy=None):

        
        super().__init__(iteration_limit=iteration_limit,
                         rollout_policy= None,
                         evaluation_policy=evaluation_policy,
                         rollout_repetition = 1,
                         environment_id='splendor-v0')


        self.exploration_parameter = exploration_parameter
        self.score_evaluator = UCB1Score(self.exploration_parameter)
        self.root = None
        self.path_from_original_root = None

    def create_root(self, observation: DeterministicObservation):
        self.original_root = DeterministicTreeNode(observation, parent=None, parent_action=None, reward=0, is_done=False,winner_id=None)
        self.root = self.original_root
        self.path_from_original_root = [self.original_root]

    def change_root(self, node):
        self.root = node

    def return_root(self):
        return self.root

    def return_original_root(self):
        return self.original_root

    def _evaluate_leaf(self, leaf: DeterministicTreeNode):
        self.env.load_observation(leaf.observation)
        evaluated_player_id = self.env.current_state_of_the_game.active_player_id
        return evaluated_player_id, self.evaluation_policy.evaluate_state(self.env.current_state_of_the_game)

    # def _evaluate(self, observation: DeterministicObservation):
    #     self.env.load_observation(observation)
    #     evaluated_player_id = self.env.current_state_of_the_game.active_player_id
    #     return evaluated_player_id, self.evaluation_policy.evaluate_state(self.env.current_state_of_the_game)


    def move_root(self, action):
        if self.root.expanded() == False:
            self._expand_leaf(self.root)
        self.root = self.root.action_to_children_dict[action.__repr__()]
        self.path_from_original_root.append(self.root)

    def _tree_traversal(self):
        search_path = [self.root]
        while search_path[-1].expanded() and not search_path[-1].terminal:
            node_to_add = self._select_child(search_path[-1])
            search_path.append(node_to_add)

        return search_path[-1], search_path

    def _expand_leaf(self, leaf: DeterministicTreeNode):
        terminal_children = []
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
                if new_child.terminal:
                    terminal_children.append(new_child)
                #else:
                    #new_child.value_acc.add(self.evaluation_policy.evaluate_state(new_child.observation.recreate_state()))
                # if not new_child.teminal:
                #     child_player_id, child_value = self._evaluate(child_state_observation)
                #     new_child.value_acc.add(self._evaluate(child_value))
        return terminal_children


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
                pass
                #print('\n WARNING: MCTS has not evaluated all possible moves. Choosing from a subset. \n')
            return node.children[best_child_index], node.actions[best_child_index]
        else:
            #print(print('\n WARNING: MCTS has not evaluated all possible moves. Choosing from a subset. \n'))
            return None, None


    def _backpropagate(self, search_path: List[DeterministicTreeNode], value, eval_id, to_original_root: bool = True):
        if to_original_root:
            search_path = self.path_from_original_root[:-1] + search_path
        for node in search_path:
            if node.active_player_id() == eval_id:
                node.value_acc.add(-value)
            else:
                node.value_acc.add(value)


    def _backpropagate_evaluation(self, search_path: List[DeterministicTreeNode], evaluated_player_id, value):
        assert evaluated_player_id is not None, 'Provide id of evaluated player'
        for node in search_path:
            if node.active_player_id() == evaluated_player_id:
                node.value_acc.add(value)
            else:
                node.value_acc.add(-value)

    def choose_action(self):
        _, best_action = self._select_best_child()
        return best_action


    def reverse_search_path(self, search_path: List[DeterministicTreeNode]):
        reversed_search_path = []
        for node in search_path:
            reversed_search_path = [node] + reversed_search_path
        return reversed_search_path

    def run_mcts_pass(self):
        current_leaf, tree_path = self._tree_traversal()
        terminal_children = self._expand_leaf(current_leaf)
        terminal_player_id = (self.env.current_state_of_the_game.active_player_id+1)%2
        for terminal_child in terminal_children:
            self._backpropagate(tree_path, terminal_child.value_acc.get(), terminal_player_id)
        if current_leaf.value_acc._count == 0:
            id, val = self._evaluate_leaf(current_leaf)
            current_leaf.value_acc.add(val)
            self._backpropagate(tree_path, val, id)
        else:
            current_leaf.check_if_terminal()
            if not current_leaf.terminal:
                new_leaf = self._select_child(current_leaf)
                id, val = self._evaluate_leaf(new_leaf)
                tree_path.append(new_leaf)
                self._backpropagate(tree_path, val, id)

    def run_simulation(self, n_passes):
        for _ in range(n_passes):
            self.run_mcts_pass()