from agents.greedy_agent_boost import GreedyAgentBoost
from agents.greedysearch_agent import GreedySearchAgent
from agents.random_agent import RandomAgent
from arena.arena import Arena
from arena.arena_multi_thread import ArenaMultiThread
from arena.game_statistics_duels import GameStatisticsDuels
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.state import State
import pandas as pd

from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
main_process = my_rank == 0

class Judger:

    def __init__(self, n_repetitions):
        self.local_arena = Arena()


    def judge_dataframe(self, filename, n_repetitions):
        color = my_rank
        name = 'col_{}_'.format(color) + filename
        df = pd.read_pickle(name + '.pi')
        judged_df = pd.DataFrame()
        for index, row in df.iterrows():
            observation = row['observation']
            results, best_action = self.judge_observation(observation, n_repetitions = 5)
            n_games, reward, _ = results.return_stats()
            value = reward / n_games
            judged_df = judged_df.append({'observation' : observation, 'value' : value, 'best_action': best_action}, ignore_index=True)

        judged_df.to_pickle(name + '_judged.pi')
        judged_df.to_csv(name + '_judged.csv')


    def judge_observation(self, observation : DeterministicObservation, n_repetitions : int):

        active_agent = GreedySearchAgent()
        active_agent.name = 'Active'
        opp_agent = GreedySearchAgent()
        opp_agent.name = 'Other'

        # active_agent = GreedyAgentBoost(distribution='first_buy')
        # opp_agent = GreedyAgentBoost(distribution='first_buy')

        #jobs per process:
        # jobs_per_process = int(n_repetitions / comm.Get_size())
        # remaining_jobs = n_repetitions % comm.Get_size()
        # print('jobs per proc = {}'.format(jobs_per_process))
        # print('remaining = {}'.format(remaining_jobs))

        results = results = self.local_arena.run_many_duels('deterministic', [active_agent, opp_agent],
                                                   number_of_games=n_repetitions, shuffle_agents=False,
                                                   starting_agent_id=0, initial_observation=observation)

        # if my_rank < remaining_jobs:
        #     results = self.local_arena.run_many_duels('deterministic', [active_agent, opp_agent],
        #                                           number_of_games=jobs_per_process+1, shuffle_agents=False,
        #                                           starting_agent_id=0, initial_observation=observation)
        # if my_rank >= remaining_jobs:
        #     if jobs_per_process > 0:
        #         results = self.local_arena.run_many_duels('deterministic', [active_agent, opp_agent],
        #                                                   number_of_games=jobs_per_process, shuffle_agents=False,
        #                                                   starting_agent_id=0, initial_observation=observation)
        #     else:
        #         results = None
        #
        # combined_results = comm.gather(results, root=0)
        # if main_process:
        #     new_results = GameStatisticsDuels([active_agent, opp_agent])
        #     for one_process_results in combined_results:
        #         new_results.register(one_process_results)
        #
        # if not main_process:
        #     new_results = None

        best_action = active_agent.choose_action(observation, previous_actions=[None])

        return results, best_action





