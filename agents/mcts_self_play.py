from agents.multi_process_mcts_agent import MultiProcessMCTSAgent
from monte_carlo_tree_search.evaluation_policies.abstract_evaluation_policy import EvaluationPolicy
from monte_carlo_tree_search.rollout_policies.abstract_rolluot_policy import RolloutPolicy


class MCTSSelfPlay(MultiProcessMCTSAgent):

    def __init__(self,
                 iteration_limit,
                 evaluation_policy: EvaluationPolicy = None,
                 create_visualizer: bool = True,
                 show_unvisited_nodes=False):

        super().__init__(iteration_limit=iteration_limit, rollout_policy=None, evaluation_policy=evaluation_policy,
                         rollout_repetition=1, create_visualizer=create_visualizer, show_unvisited_nodes=show_unvisited_nodes)


        