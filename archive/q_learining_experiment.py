import time

from agents.greedy_agent_boost import GreedyAgentBoost
from archive.q_learning import MultiQLearningTrainer

opponent_x = GreedyAgentBoost()
#opponent_x = RandomAgent(distribution='first_buy')
fufu = MultiQLearningTrainer(alpha=0.1)
t_start = time.time()
fufu.run_full_training(n_iterations=150, opponent=opponent_x)
t_end = time.time()
print('Timne for 2000 iterations = {}'.format(t_end - t_start))