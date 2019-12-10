import random

from mpi4py import MPI
from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.random_agent import RandomAgent
from archive.simple_mcts_agent import SimpleMCTSAgent
from arena.multi_arena import MultiProcessSingleDuelArena

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0


agent1 = RandomAgent(mpi_communicator=comm)
agent2 = RandomAgent(mpi_communicator=comm)
#
agent3  = MultiProcessMCTSAgent(150, 5, True, False)
agent4 = SimpleMCTSAgent(10)

random.seed(2)

# random.randint.seed(100)
arek = MultiProcessSingleDuelArena()
result = arek.run_one_duel_multi_process_deterministic(comm, [agent3, agent1])

if main_process:
    print(result)

#fufu = MultiMCTS(comm)

#fufu.create_root(State())

#fufu.run_simulation(15, 5)

# arek = Arena()
# result = arek.run_one_duel([agent1, agent4])













#
# my_color = my_rank % 2
# NEW_COMM = MPI.COMM_WORLD.Split(my_color)
#
# fuf = MultiMCTS(comm)
# #fuf.execute()
#
# dup = fuf.give_random_number()
#
# tutu = comm.gather(dup, root=0)
#
# if main_process:
#     print(tutu)
#
# fuf = MultiMCTS(NEW_COMM)
# stanek = State()
# fuf.create_root(stanek)
#
# if my_rank == 0:
#     start = time.time()
#
# fuf.run_simulation(100, 1)
# if my_rank == 0:
#     print('Time taken = {}'.format(time.time() - start))
#
# if main_process:
#     tit = TreeVisualizer(show_unvisited_nodes=True)
#     tit.generate_html(fuf.return_root(), 'dudik.html')