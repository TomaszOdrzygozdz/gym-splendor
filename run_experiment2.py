import time

from mpi4py import MPI

from agents.multi_process_mcts_agent import MultiMCTSAgent
from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout

my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='first_buy')
agent1a = RandomAgent(distribution='first_buy')
#agentG = GreedyAgentBoost()
#agent2 = DenseNNAgent(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\minmax_480_games.h5')
#agent3 = MultiMCTSAgent(250, evaluation_policy=ValueEvaluator(), create_visualizer=True)
agent4 = MultiMCTSAgent(5, rollout_policy=RandomRollout(distribution='first_buy'), rollout_repetition=2, create_visualizer=True)

# agent5 = MultiMCTSAgent(3, 5, True)
# agent6 = GeneralMultiProcessMCTSAgent(10, 2, True, False,
#                                         mcts = "rollout",
#                                         param_1 = "random",
#                                         param_2 = "uniform")
#agent7 = ValueNNAgent(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\\value_random_rollout_960.h5')


arek = MultiArena()
time_s = time.time()

# import cProfile
# pro = cProfile.Profile()

#wyn = pro.run('arek.run_many_duels(\'deterministic\', [agent3, agent4], n_games=1, n_proc_per_agent=10)')
#wyn.dump_stats('stat.prof')
fuf = arek.run_many_duels('deterministic', [agent1, agent4], n_games=1, n_proc_per_agent=10)

#arek.run_multi_process_self_play('deterministic', agent4, render_game=False)

if main_process:
    print(fuf)
    print('Time = {}'.format(time.time() - time_s))
    #data_collecting
    # root = agent3.mcts_algorithm.original_root()
    # data_collector = TreeDataCollector(root)
    # data_collector.dump_data(file_name='E:\ML_research\gym_splendor\\nn_models\data\\tree_data')


