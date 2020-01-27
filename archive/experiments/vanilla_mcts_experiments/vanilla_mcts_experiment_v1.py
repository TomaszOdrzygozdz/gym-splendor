from mpi4py import MPI
from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import DeterministicMultiProcessArena

comm = MPI.COMM_WORLD
my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

def run():
    agent1 = RandomAgent(mpi_communicator=comm)
    agent2 = RandomAgent(mpi_communicator=comm)
    #
    agent3  = MultiMCTSAgent(1, 5, True, False)

    # random.randint.seed(100)
    arek = DeterministicMultiProcessArena()
    result = arek.run_one_duel_multi_process_deterministic(comm, [agent3, agent1])
    result2 = arek.run_one_duel_multi_process_deterministic(comm, [agent3, agent1])

    if main_process:
        print(result)
        print(result2)