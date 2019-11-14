from agent import Agent
from gym_splendor_code.envs.mechanics.state import State
from mcts_alogrithms.lukas_mcts import MCTSValue
from mcts_alogrithms.lukas_mcts.mcts_env_model import SplendorModelMCTS
from mcts_alogrithms.lukas_mcts.value_trainable import ValueZero


class PoloPlusMCTS(Agent):

    def __init__(self, episode_max_steps=10):
        super().__init__()
        self.model = SplendorModelMCTS(self.env)
        self.mcts_algorithm = MCTSValue(self.model,
                                        episode_max_steps=episode_max_steps,
                                        value=ValueZero(),
                                        node_value_mode='bootstrap')

    def choose_action(self, observation):
        self.mcts_algorithm.run_one_episode()
        _, _, action = self.mcts_algorithm.run_one_step(observation)
        return action

state = State()
lu = PoloPlusMCTS()
print(lu.choose_action(state.jsonize()))
