import time

from mpi4py import MPI

from agents.dense_nn_agent import DenseNNAgent
from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from agents.random_agent import RandomAgent
from agents.value_nn_agent import ValueNNAgent
from arena.arena import Arena
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.evaluation_policies.dummy_eval import DummyEval
from monte_carlo_tree_search.evaluation_policies.nn_evaluation import QValueEvaluator
from monte_carlo_tree_search.evaluation_policies.value_evaluation_nn import ValueEvaluator
from monte_carlo_tree_search.rollout_policies.dense_nn_rollout import DenseNNRollout
from monte_carlo_tree_search.rollout_policies.greedy_rollout import GreedyRolloutPolicy
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout
from nn_models.tree_data_collector import TreeDataCollector

my_rank = MPI.COMM_WORLD.Get_rank()
main_process = my_rank==0

agent1 = RandomAgent(distribution='first_buy')
#agent2 = DenseNNAgent(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\minmax_480_games.h5')
#agent3 = MultiProcessMCTSAgent(250, evaluation_policy=ValueEvaluator(), create_visualizer=True)
agent4 = MultiProcessMCTSAgent(5, rollout_policy=RandomRollout(distribution='first_buy'), rollout_repetition=2, create_visualizer=True)

# agent5 = MultiProcessMCTSAgent(3, 5, True)
# agent6 = GeneralMultiProcessMCTSAgent(10, 2, True, False,
#                                         mcts = "rollout",
#                                         param_1 = "random",
#                                         param_2 = "uniform")
agent7 = ValueNNAgent(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\\value_random_rollout_960.h5')


arek = MultiArena()
time_s = time.time()

# import cProfile
# pro = cProfile.Profile()

#wyn = pro.run('arek.run_many_duels(\'deterministic\', [agent3, agent4], n_games=1, n_proc_per_agent=10)')
#wyn.dump_stats('stat.prof')
#fuf = arek.run_many_duels('deterministic', [agent1], n_games=10, n_proc_per_agent=1)
fuf = 0

arek.run_multi_process_self_play('deterministic', agent4, render_game=False)

if main_process:
    print(fuf)
    print('Time = {}'.format(time.time() - time_s))
    #data_collecting
    # root = agent3.mcts_algorithm.original_root()
    # data_collector = TreeDataCollector(root)
    # data_collector.dump_data(file_name='E:\ML_research\gym_splendor\\nn_models\data\\tree_data')


