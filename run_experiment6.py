from agents.multi_process_mcts_agent import MultiMCTSAgent
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from monte_carlo_tree_search.mcts_algorithms.multi_process.multi_mcts import MultiMCTS
from monte_carlo_tree_search.rollout_policies.random_rollout import RandomRollout

from mpi4py import MPI

from monte_carlo_tree_search.self_play_trainer import SelfPlayTrainer

comm = MPI.COMM_WORLD

trainer = SelfPlayTrainer('dqn', 10, 2, 0.5)
trainer.prepare_training()
trainer.full_training(n_repetitions=2, alpha=0.05, epochs=1)
