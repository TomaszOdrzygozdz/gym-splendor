import gin
from nn_models.architectures.average_pool_v0 import StateEvaluator
gin.parse_config_file('/home/tomasz/ML_Research/splendor/gym-splendor/nn_models/experiments/series_1/experiment_1/params_v1.gin')

from agents.random_agent import RandomAgent
from agents.value_nn_agent import ValueNNAgent
from arena.arena import Arena


arek = Arena()

agent1 = ValueNNAgent()
agent2 = RandomAgent()

result = arek.run_many_duels('deterministic', [agent1, agent2], 2, True)
print(result)
_, reward, wins  = result.return_stats()
print(wins)
