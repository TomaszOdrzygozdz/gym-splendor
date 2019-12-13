from agents.multi_process_mcts_agent import MultiMCTSAgent
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from monte_carlo_tree_search.mcts_algorithms.multi_process.multi_mcts import MultiMCTS
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout

from mpi4py import MPI
comm = MPI.COMM_WORLD

ag = MultiMCTSAgent(iteration_limit=3, only_best=0.1, rollout_policy=RandomRollout(), evaluation_policy=ValueEvaluator(weights_file=None),
                    rollout_repetition=2, create_visualizer=True, show_unvisited_nodes=False)


stanek = State()
obek = DeterministicObservation(stanek)
print(type(comm))
ag.set_communicator(comm)
ag.initialize_mcts(mpi_communicator=comm)
print(ag.choose_action(obek, [None]))
ag.draw_final_tree()

