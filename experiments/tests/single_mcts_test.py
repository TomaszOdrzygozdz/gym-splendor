import gin

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer
gin.parse_config_file('/home/tomasz/ML_Research/splendor/gym-splendor/experiments/MCTS_series_1/params.gin')

from agents.random_agent import RandomAgent
from agents.single_mcts_agent import SingleMCTSAgent
from arena.arena import Arena
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from monte_carlo_tree_search.mcts_algorithms.single_process.single_mcts import SingleMCTS

arek = Arena()

a1 = RandomAgent()
a2 = SingleMCTSAgent(5, ValueEvaluator(), 0.6, True, True)
#
results = arek.run_one_duel('deterministic', [a1, a2])

# state1 = State()
# fufu = SingleMCTS(5, 0.6,  ValueEvaluator())
# fufu.create_root(DeterministicObservation(state1))
# fufu.run_mcts_pass()