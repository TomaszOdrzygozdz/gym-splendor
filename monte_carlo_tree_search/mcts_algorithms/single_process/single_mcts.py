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
                 exploration_parameter = 1/math.sqrt(6),
                 rollout_policy=None,
                 evaluation_policy=None,
                 rollout_repetition = 10):

        
        super().__init__(iteration_limit=iteration_limit,
                         rollout_policy= rollout_policy,
                         evaluation_policy=evaluation_policy,
                         rollout_repetition = 1,
                         environment_id='splendor-v0')


        self.exploration_parameter = exploration_parameter
        self.rollout_policy = rollout_policy
        self.score_evaluator = UCB1Score(self.exploration_parameter)
        self.root = None

    def create_root(self, observation: DeterministicObservation):
        self.original_root = DeterministicTreeNode(observation, parent=None, parent_action=None, reward=0, is_done=False,winner_id=None)
        self.root = self.original_root

    def change_root(self, node):
        self.root = node

    def return_root(self):
        return self.root

    def return_original_root(self):
        return self.original_root

    def _rollout(self, observation: DeterministicObservation):
        value = 0
        is_done = False
        assert observation is not None, 'Observation is None'
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

    def _evaluate(self, observation: DeterministicObservation):
        self.env.load_observation(observation)
        evaluated_player_id = self.env.current_state_of_the_game.active_player_id
        return evaluated_player_id, self.evaluation_policy.evaluate_state(self.env.current_state_of_the_game)

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

    # def _expand_leaf_with_eval(self, leaf: DeterministicTreeNode):
    #     if not leaf.expanded():
    #         leaf.generate_actions()
    #         leaf.check_if_terminal()
    #         evaluated_player = leaf.state.active_player_id
    #         if leaf.actions:
    #             predicted_q_values = self.evaluation_policy.evaluate_all_actions(leaf.state, leaf.actions)
    #         for action_id, action in enumerate(leaf.actions):
    #             self.env.load_observation(leaf.observation)
    #             child_state_observation, reward, is_done, info = self.env.step('deterministic', action)
    #             winner_id = info['winner_id']
    #             new_child = DeterministicTreeNode(child_state_observation, leaf, action, reward, is_done, winner_id)
    #             leaf.action_to_children_dict[action.__repr__()] = new_child
    #             leaf.children.append(new_child)
    #             new_child.value_acc.add(predicted_q_values[action_id])
    #         return evaluated_player, predicted_q_values

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


    def _backpropagate(self, search_path: List[DeterministicTreeNode], winner_id, value):
        for node in search_path:
            if winner_id is not None:
                if node.active_player_id() == winner_id:
                    node.value_acc.add(-value)
                else:
                    node.value_acc.add(value)
            else:
                node.value_acc.add(0)

    def _backpropagate_evaluation(self, search_path: List[DeterministicTreeNode], evaluated_player_id, value):
        assert evaluated_player_id is not None, 'Provide id of evaluated player'
        for node in search_path:
            if node.active_player_id() == evaluated_player_id:
                node.value_acc.add_evaluation(value)
            else:
                node.value_acc.add_evaluation(-value)

    def choose_action(self):
        _, best_action = self._select_best_child()
        return best_action
