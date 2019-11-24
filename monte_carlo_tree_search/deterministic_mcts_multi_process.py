from typing import List
# from mpi4py import MPI
from tqdm import tqdm

from monte_carlo_tree_search.mcts import MCTS
from monte_carlo_tree_search.mcts_algorithms.deterministic_vanilla_mcts import DeterministicVanillaMCTS
from monte_carlo_tree_search.rolluot_policy import RolloutPolicy

# comm = MPI.COMM_WORLD
# my_rank = MPI.COMM_WORLD.Get_rank()
# main_thread = my_rank==0
from monte_carlo_tree_search.tree import TreeNode
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode

class DeterministicMCTSMultiProcess:
    def __init__(self,
                 mpi_communicator,
                 iteration_limit: int = 1000,
                 rollout_policy: RolloutPolicy = 0,
                 rollout_repetition: int = 0,
                 environment_id: str = 0) -> None:

        self.mpi_communicator = mpi_communicator
        self.my_rank = self.mpi_communicator.Get_rank()
        self.my_comm_size = mpi_communicator.Get_size()
        self.main_process = self.my_rank == 0
        self.mcts = DeterministicVanillaMCTS(iteration_limit=iteration_limit)
        self.iterations_done_so_far = 0

    #method needed only for main thread:
    def create_root(self, state_or_observation):
        if self.main_process:
            self.mcts.create_root(state_or_observation)
        else:
            pass


    def prepare_list_of_states_to_rollout(self, leaf: DeterministicTreeNode, iteration_limit: int):
        assert leaf.expanded(), 'Leaf is not yet expanded'
        children = leaf.children
        n_child_to_rollout = min(len(children), iteration_limit)
        #print('n_child_to_rollout = {}'.format(n_child_to_rollout))
        childs_per_process = int(n_child_to_rollout/ self.my_comm_size)
        #print('childs_per_process = {}'.format(childs_per_process))
        remaining = n_child_to_rollout%self.my_comm_size
        #print('remaining = {}'.format(remaining))
        states_to_rollout = []
        #print('my_comm_size = {}'.format(self.my_comm_size))
        for process_number in range(self.my_comm_size):
            if process_number < remaining:
                states_for_i_th_process = {i*self.my_comm_size + process_number: children[i*self.my_comm_size + process_number] for i in range(0,childs_per_process + 1)}
                states_to_rollout.append(states_for_i_th_process)
               # print('Process {}  got {} states to rollout'.format(process_number, len(states_for_i_th_process)))
            if process_number >= remaining and process_number < n_child_to_rollout:
                states_for_i_th_process = {i * self.my_comm_size + process_number: children[i * self.my_comm_size + process_number] for i in
                                           range(0, childs_per_process)}
                states_to_rollout.append(states_for_i_th_process)
                #print('Process {}  got {} states to rollout'.format(process_number, len(states_for_i_th_process)))
            if process_number >= n_child_to_rollout:
                states_to_rollout.append({})

        return states_to_rollout, n_child_to_rollout

    def run_mcts_pass(self, iteration_limit):

        if self.main_process:
            leaf, search_path = self.mcts._tree_traversal()
            self.mcts._expand_leaf(leaf)
            #rollout_repetition = self.n_rollout_repetition if rollout_repetition is None else rollout_repetition

        iteration_limit_for_expand = iteration_limit - self.iterations_done_so_far

        states_to_rollout = None
        jobs_to_do = None
        if self.main_process:
            states_to_rollout, jobs_to_do = self.prepare_list_of_states_to_rollout(leaf, iteration_limit_for_expand)

        jobs_done= self.mpi_communicator.bcast(jobs_to_do, root=0)


        my_nodes_to_rollout = self.mpi_communicator.scatter(states_to_rollout, root=0)
        my_results = self._rollout_many_nodes(my_nodes_to_rollout)

        combined_results = self.mpi_communicator.gather(my_results, root=0)
        #if self.main_process:
        if self.main_process:
            flattened_results = self.flatten_list_of_dicts(combined_results)
            self._backpropagate_many_results(search_path, flattened_results)

        return jobs_done

    def _rollout_many_nodes(self, dict_of_nodes):
        rollout_results_dict = {}
        if len(dict_of_nodes) > 0:
            for i in dict_of_nodes:
                winner_id, value = self.mcts._rollout(dict_of_nodes[i])
                rollout_results_dict[i] = (winner_id, value)
        return rollout_results_dict

    def _backpropagate_many_results(self, search_path, rollout_results):
        for i in rollout_results:
            this_child = search_path[-1].children[i]
            this_particulat_search_path = search_path + [this_child]
            #print(this_particulat_search_path)
            winner_id, value = rollout_results[i]
            self.mcts._backpropagate(this_particulat_search_path, winner_id, value)


    def flatten_list_of_dicts(self, list_of_dicts):
        combined_dict = {}
        for rollout_dict in list_of_dicts:
            combined_dict.update(rollout_dict)
        return combined_dict

    def run_simulation(self, iteration_limit):
        iterations_done_so_far = 0
        while iterations_done_so_far < iteration_limit:
            if self.main_process:
                print('HEJ DOING PASS')
            limit_for_this_pass = iteration_limit - iterations_done_so_far
            if self.main_process:
                print(limit_for_this_pass)
            jobs_done = self.run_mcts_pass(limit_for_this_pass)
            iterations_done_so_far += jobs_done
            if self.main_process:
                print('DONE SO FAR {}'.format(iterations_done_so_far))

    def return_root(self):
        return self.mcts.root

