from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State


class Judger:

    def __init__(self, n_repetitions):
        self.n_repetitions = n_repetitions
        self.local_arena = Arena()

    def judge_observation(self, observation : DeterministicObservation):

        active_agent = GreedyAgentBoost()
        active_agent.name = 'Active'
        opp_agent = GreedyAgentBoost()
        opp_agent.name = 'Other'

        # active_agent = GreedyAgentBoost(distribution='first_buy')
        # opp_agent = GreedyAgentBoost(distribution='first_buy')

        results1 = self.local_arena.run_many_duels('deterministic', [active_agent, opp_agent],
                                                  number_of_games=self.n_repetitions, shuffle_agents=False,
                                                  starting_agent_id=0, initial_observation=observation)


        print(results1)


