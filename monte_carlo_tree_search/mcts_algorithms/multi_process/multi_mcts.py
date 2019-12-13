# from mpi4py import MPI

import numpy as np

from gym_splendor_code.envs.mechanics.game_settings import USE_TQDM
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode

if USE_TQDM:
    from tqdm import tqdm

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.action import Action
from monte_carlo_tree_search.mcts_algorithms.single_process.single_mcts import SingleMCTS
from monte_carlo_tree_search.rollout_policies.abstract_rolluot_policy import RolloutPolicy

# comm = MPI.COMM_WORLD
# my_rank = MPI.COMM_WORLD.Get_rank()
# main_thread = my_rank==0

class MultiMCTS:
    def __init__(self,
                 mpi_communicator,
                 iteration_limit: int = 1000,
                 rollout_policy: RolloutPolicy = RandomRollout(distribution='first_buy'),
                 evaluation_policy: EvaluationPolicy = None,
                 rollout_repetition: int = 0,
                 environment_id: str = 0) -> None:

        self.mpi_communicator = mpi_communicator
        self.my_rank = self.mpi_communicator.Get_rank()
        self.my_comm_size = mpi_communicator.Get_size()
        self.main_process = self.my_rank == 0
        self.mcts = SingleMCTS(iteration_limit=iteration_limit, rollout_policy=rollout_policy,
                               evaluation_policy=evaluation_policy, rollout_repetition=rollout_repetition)
        self.iterations_done_so_far = 0

    #method needed only for main thread:
    def create_root(self, observation : DeterministicObservation):
        if self.main_process:
            self.mcts.create_root(observation)
        else:
            pass


    def prepare_list_of_states_to_rollout(self, leaf: DeterministicTreeNode, iteration_limit: int, choose_best=None):
        assert leaf.expanded(), 'Leaf is not yet expanded'

        children = leaf.children
        not_terminal_children = [child for child in children if child.terminal == False]
        terminal_children = [child for child in children if child.terminal == True]

        n_child_to_rollout = min(len(not_terminal_children), iteration_limit)
        childs_per_process = int(n_child_to_rollout/ self.my_comm_size)

        states_to_rollout = []

        if choose_best is not None and iteration_limit > 1:
            k_max = max(int(n_child_to_rollout*choose_best),1)
            if k_max > 1:
                #read nodes evaluations:
                nodes_list = []
                nodes_values_list = []

                for node in not_terminal_children:
                    if node.value_acc.get() is not None:
                        nodes_list.append(node)
                        nodes_values_list.append(node.value_acc.get())

                max_ind = list(np.argpartition(nodes_values_list, k_max)[-k_max:])
                max_nodes_list = []
                max_values = []
                for idx in max_ind:
                    max_nodes_list.append(nodes_list[idx])
                    max_values.append(nodes_values_list[idx])
                not_terminal_children = max_nodes_list
                n_child_to_rollout = len(max_nodes_list)
                childs_per_process = int(n_child_to_rollout / self.my_comm_size)

        remaining = n_child_to_rollout % self.my_comm_size

        for process_number in range(self.my_comm_size):
            if process_number < remaining:
                states_for_i_th_process = {i*self.my_comm_size + process_number: not_terminal_children[i*self.my_comm_size + process_number].observation for i in range(0,childs_per_process + 1)}
                states_to_rollout.append(states_for_i_th_process)
            if process_number >= remaining and process_number < n_child_to_rollout:
                states_for_i_th_process = {i * self.my_comm_size + process_number: not_terminal_children[i * self.my_comm_size + process_number].observation for i in
                                           range(0, childs_per_process)}
                states_to_rollout.append(states_for_i_th_process)
            if process_number >= n_child_to_rollout:
                states_to_rollout.append({})


        return terminal_children, states_to_rollout, n_child_to_rollout


    def run_mcts_pass(self, iteration_limit, rollout_repetition, choose_best):

        if self.main_process:
            leaf, search_path = self.mcts._tree_traversal()
            self.mcts._expand_leaf(leaf)

        iteration_limit_for_expand = iteration_limit - self.iterations_done_so_far

        states_to_rollout = None
        jobs_to_do = None
        if self.main_process:
            terminal_children, states_to_rollout, jobs_to_do = self.prepare_list_of_states_to_rollout(leaf, iteration_limit_for_expand, choose_best=None)

        jobs_done= self.mpi_communicator.bcast(jobs_to_do, root=0)
        my_nodes_to_rollout = self.mpi_communicator.scatter(states_to_rollout, root=0)

        #first eval nodes
        if self.mcts.tree_mode == 'evaluation' or self.mcts.tree_mode == 'combined':
            my_results = self._evaluate_many_nodes(my_nodes_to_rollout)
            combined_results = self.mpi_communicator.gather(my_results, root=0)
            if self.main_process:
                flattened_results = self.flatten_list_of_dicts(combined_results)
                if self.mcts.tree_mode == 'evaluation' or self.mcts.tree_mode == 'combined':
                    self._backpropagate_many_results('evaluation', search_path, flattened_results)

        #now rollout nodes:
        if self.mcts.tree_mode == 'rollout':
            for _ in range(rollout_repetition):
                my_results = self._rollout_many_nodes(my_nodes_to_rollout)

        if self.mcts.tree_mode == 'combined':
            if self.main_process:
                _, states_to_rollout, jobs_to_do = self.prepare_list_of_states_to_rollout(leaf,
                                                                                                          iteration_limit_for_expand,
                                                                                                          choose_best=choose_best)
            jobs_done = self.mpi_communicator.bcast(jobs_to_do, root=0)
            my_nodes_to_rollout = self.mpi_communicator.scatter(states_to_rollout, root=0)
            for _ in range(rollout_repetition):
                my_results = self._rollout_many_nodes(my_nodes_to_rollout)


        combined_results = self.mpi_communicator.gather(my_results, root=0)
        #if self.main_process:
        if self.main_process:
            flattened_results = self.flatten_list_of_dicts(combined_results)
            if self.mcts.tree_mode == 'rollout' or self.mcts.tree_mode == 'combined':
                self._backpropagate_many_results('rollout', search_path, flattened_results)

        #colloect values for terminal children:
        if self.main_process:
            for terminal_child in terminal_children:
                for _ in range(rollout_repetition):
                    value = 0
                    winner_id = leaf.winner_id
                    if leaf.perfect_value is not None:
                        value = leaf.perfect_value
                        local_search_path = search_path + [terminal_child]
                        self.mcts._backpropagate('rollout', local_search_path, winner_id, value)

        return jobs_done

    def _rollout_many_nodes(self, dict_of_states):
        rollout_results_dict = {}
        if len(dict_of_states) > 0:
            for i in dict_of_states:
                winner_id, value = self.mcts._rollout(dict_of_states[i])
                rollout_results_dict[i] = (winner_id, value)
        return rollout_results_dict

    def _evaluate_many_nodes(self, dict_of_states):
        evaluation_results_dict = {}
        if len(dict_of_states) > 0:
            for i in dict_of_states:
                evaluated_player_id, value = self.mcts._evaluate(dict_of_states[i])
                evaluation_results_dict[i] = (evaluated_player_id, value)
        return evaluation_results_dict

    def _backpropagate_many_results(self, backprop_mode, search_path, rollout_results):
        for i in rollout_results:
            this_child = search_path[-1].children[i]
            this_particular_search_path = search_path + [this_child]
            if backprop_mode == 'rollout':
                winner_id, value = rollout_results[i]
                self.mcts._backpropagate(this_particular_search_path, winner_id, value)
            if backprop_mode == 'evaluation':
                evaluated_player_id, value = rollout_results[i]
                self.mcts._backpropagate_evaluation(this_particular_search_path, evaluated_player_id, value)

    def flatten_list_of_dicts(self, list_of_dicts):
        combined_dict = {}
        for rollout_dict in list_of_dicts:
            combined_dict.update(rollout_dict)
        return combined_dict

    def move_root(self, action : Action):
        if self.main_process:
            self.mcts.move_root(action)
        else:
            pass

    def original_root(self):
        return self.mcts.original_root

    def choose_action(self):
        if self.main_process:
            return self.mcts.choose_action()
        else:
            return None

    def create_progress_bar(self, lenght):
        if self.main_process:
            if USE_TQDM:
                self.progress_bar = tqdm(total=lenght, postfix=None)

    def set_progress_bar(self, value):
        if self.main_process:
            self.progress_bar.n = min(value, self.progress_bar.total-1)
            self.progress_bar.update()

    def run_simulation(self, iteration_limit, rollout_repetition, only_best):

        iterations_done_so_far = 0
        while iterations_done_so_far < iteration_limit:
            limit_for_this_pass = iteration_limit - iterations_done_so_far
            jobs_done = self.run_mcts_pass(limit_for_this_pass, rollout_repetition, only_best)
            if jobs_done == 0:
                break
            iterations_done_so_far += jobs_done

    def return_root(self):
        return self.mcts.root


