from agent import Agent
from monte_carlo_tree_search.mcts_algorithms.deterministic_vanilla_mcts import DeterministicVanillaMCTS
from monte_carlo_tree_search.tree_visualizer.tree_visualizer import TreeVisualizer




class SimpleMCTSAgent(Agent):

    action_number = 0

    def __init__(self,
                 iteration_limit = 10):

        super().__init__()
        self.mcts_algorithm = DeterministicVanillaMCTS(iteration_limit=iteration_limit)
        self.mcts_started = False
        # we create own gym-splendor enivronemt to have access to its functionality
        # We specify the name of the agent
        self.name = 'MCTS'
        self.visualizer = TreeVisualizer(show_unvisited_nodes=False)


    def choose_action(self, observation, previous_actions):
        state_to_eval = self.env.observation_space.observation_to_state(observation)
        self.mcts_algorithm.create_root(state_to_eval)
        self.mcts_algorithm.run_simulation()
        return self.mcts_algorithm._select_best_child()

    def deterministic_choose_action(self, state, previous_actions):
        opponents_action = previous_actions[0]
        if not self.mcts_started:
            self.mcts_algorithm.create_root(state)
            self.mcts_started = True
        if opponents_action is not None:
            self.mcts_algorithm.move_root(opponents_action)
        SimpleMCTSAgent.action_number+= 1
        self.mcts_algorithm.run_simulation()
        self.visualizer.generate_html(self.mcts_algorithm.root, 'renders\\action_{}.html'.format(SimpleMCTSAgent.action_number))
        best_action = self.mcts_algorithm.choose_action()
        self.mcts_algorithm.move_root(best_action)
        self.draw_final_tree()
        return best_action

    def draw_final_tree(self):
        self.visualizer.generate_html(self.mcts_algorithm.original_root, 'renders\\full_game.html')

