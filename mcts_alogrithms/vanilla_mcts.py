from typing import List, Tuple

import time
import gym
import math
import numpy as np
import random

from tqdm import tqdm

from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from mcts_alogrithms.rollout_policies.random_rollout import RandomRolloutPolicy
from mcts_alogrithms.score_compute import UCB1Score
from mcts_alogrithms.tree import TreeNode
from mcts_alogrithms.tree_renderer.tree_visualizer import TreeVisualizer

verbose = True






class FullStateVanillaMCTS:
    def __init__(self,
                 time_limit=None,
                 iteration_limit=None,
                 exploration_parameter = 1/math.sqrt(2),
                 rollout_policy=RandomRolloutPolicy(),
                 number_of_rollouts = 1):

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
        self.number_of_rollouts = number_of_rollouts
        self.score_evaluator = UCB1Score(self.exploration_parameter)

    def create_root(self, state):
        self.root = TreeNode(state=state, parent=None, parent_action=None, terminal=False)
        # self.root.generate_actions()
        # self._expand_leaf(self.root)

    def _rollout(self, leaf: TreeNode):
        is_done = False
        self.full_state_env.load_state_from_dict(leaf.state_as_dict)
        winner_id = None
        while not is_done:
            action = self.rollout_policy.choose_action(self.full_state_env.current_state_of_the_game)
            if action is not None:
                new_state, reward, is_done, winner_id = self.full_state_env.full_state_step(action)
            else:
                break
        return winner_id

    def run_mcts_pass(self):

        leaf, search_path = self._tree_traversal()
        self._expand_leaf(leaf)
        if not leaf.terminal:
            node_to_rollout = self._select_child(leaf)
            search_path.append(node_to_rollout)
            for _ in range(self.number_of_rollouts):
                winner_id = self._rollout(node_to_rollout)
                self._backpropagate(search_path, winner_id)

    def _tree_traversal(self):
 #       node = self.root
        search_path = [node]
#        new_node = self._select_child(node)

        while search_path[-1].expanded() and not search_path[-1].terminal:
            node_to_add = self._select_child(search_path[-1])
            search_path.append(node_to_add)

        return search_path[-1], search_path

    def _expand_leaf(self, leaf: TreeNode):
        # First generate all legal actions in a leaf:
        leaf.generate_actions()
        leaf.check_if_terminal()
        for action in leaf.actions:
            self.full_state_env.load_state_from_dict(leaf.state_as_dict)
            child_state, reward, is_done, who_won = self.full_state_env.full_state_step(action)
            new_child = TreeNode(child_state, leaf, action, is_done)
            leaf.children.append(new_child)


    def _select_child(self, node):
        children_ratings = [self.score_evaluator.compute_score(child, node) for child in node.children]
        child_to_choose_index = np.argmax(children_ratings)

        return node.children[child_to_choose_index]

    def _select_best_child(self, node : TreeNode=None):

        node = node if node is not None else self.root
        children_values = [child.value_acc.get() for child in node.children if child.value_acc.get() is not None]
        print([child.value_acc.get() for child in node.children])
        best_child_index = np.argmax(children_values)
        return node.children[best_child_index], node.actions[best_child_index]


    def _backpropagate(self, search_path: List[TreeNode], winner_id):
        value = 1
        for node in search_path:
            if winner_id is not None:
                if node.active_player_id() == winner_id:
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
                self.run_mcts_pass()
        else:
            for i in tqdm(range(self.searchLimit)):
                self.run_mcts_pass()

    def choose_action(self):
        _, best_action = self._select_best_child()
        return best_action



stan = State()
stan.active_players_hand().gems_possessed.gems_dict[GemColor.GREEN] = 2
stan.active_players_hand().gems_possessed.gems_dict[GemColor.BLUE] = 2
stan.active_players_hand().gems_possessed.gems_dict[GemColor.RED] = 2
stan.active_players_hand().gems_possessed.gems_dict[GemColor.BLACK] = 2
stan.active_players_hand().gems_possessed.gems_dict[GemColor.WHITE] = 2

stan.board.gems_on_board.gems_dict[GemColor.GREEN] = 2
stan.board.gems_on_board.gems_dict[GemColor.BLUE] = 2
stan.board.gems_on_board.gems_dict[GemColor.RED] = 2
stan.board.gems_on_board.gems_dict[GemColor.BLACK] = 2
stan.board.gems_on_board.gems_dict[GemColor.WHITE] = 2



mcts = FullStateVanillaMCTS(iteration_limit=50)
mcts.create_root(stan)
mcts.run_simulation()
#print(mcts._select_best_child())
#print(TreeNode.id)


vizu = TreeVisualizer(show_unvisited_nodes=False)
vizu.generate_html(mcts.root)
#




