from agents.minmax_agent import MinMaxAgent
from agents.single_mcts_agent import SingleMCTSAgent
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.evaluation_policies.value_evaluator_nn import ValueEvaluator
from nn_models.architectures.average_pool_v0 import StateEncoder, ValueRegressor, IdentityTransformer

PARAMS_FILE = 'experiments/MCTS_series_1/params.gin'
import gin
gin.parse_config_file(PARAMS_FILE)

def go():
    arena = MultiArena()
    data_transformer = IdentityTransformer()
    model = StateEncoder(final_layer=ValueRegressor(), data_transformer=data_transformer)
    model.load_weights('archive/weights_tt1/epoch_41.h5')
    value_policy = ValueEvaluator(model=model, weights_file=None)
    mcts_agent = SingleMCTSAgent(50, value_policy, 0.41,
                                              create_visualizer=False, show_unvisited_nodes=False,
                                              log_to_neptune=False)
    opp = MinMaxAgent()
    results = arena.run_many_duels('deterministic', [mcts_agent, opp], 10, 1, True)
    print(results)
