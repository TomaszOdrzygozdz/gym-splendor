import time

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.mcts_algorithms.multi_process.multi_mcts import \
    MultiMCTS

from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

mc = MultiMCTS(comm)

stan = State()
obs = DeterministicObservation(stan)

mc.create_root(obs)
t_start = time.time()
mc.run_simulation(50,1)
t_end = time.time()
if main_process:
    print('Time taken = {}'.format(t_end - t_start))