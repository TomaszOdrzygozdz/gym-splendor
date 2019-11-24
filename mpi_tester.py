from mpi4py import MPI

from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.deterministic_mcts_multi_process import DeterministicMCTSMultiProcess
from monte_carlo_tree_search.mcts_multi_process import MCTSMultiProcess
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0


# my_color = my_rank % 3
#
# NEW_COMM = MPI.COMM_WORLD.Split(my_color)
#
#
# fuf = MCTSMultiProcess(NEW_COMM)
# fuf.execute()

fuf = DeterministicMCTSMultiProcess(comm)
stanek = State()
fuf.create_root(stanek)

fuf.run_simulation(2)


if main_process:
    tit = TreeVisualizer(show_unvisited_nodes=True)
    tit.generate_html(fuf.return_root(), 'dudik.html')