from agents.greedy_agent_boost import GreedyAgentBoost
from agents.multi_process_mcts_agent import MultiMCTSAgent

from mpi4py import MPI

from agents.state_judger import Judger
from arena.multi_arena import MultiArena
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from trainers.states_list import state_4, obs4, obs3, obs2, obs1, obs1_1, obs1_2

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
print(my_rank)


from agents.random_agent import RandomAgent
from arena.arena_multi_thread import ArenaMultiThread


# multi_arena = ArenaMultiThread()
#
agent1 = RandomAgent(distribution = 'uniform')
agent2 = MultiMCTSAgent(5, None, RandomRollout())
agent3 = GreedyAgentBoost()
# agent3 = RandomAgent(distribution = 'first_buy')
#
#
# multi_arena.start_collecting_states()
# results = multi_arena.all_vs_all('deterministic', [agent1, agent2, agent3], n_games=2)



# multi_arena = MultiArena()
# multi_arena.run_many_duels('deterministic', [agent2, agent3], 1, 5, False, )

fufix = Judger(50)

x = obs1
fufix.judge_observation(x)
# r = []
# for i in range(50):
#     fufix.judge_observation(x)
#     r =

