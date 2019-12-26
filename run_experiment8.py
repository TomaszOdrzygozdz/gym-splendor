
# from alpaca.alpacka.agents import DeterministicMCTSAgent
# from gym_splendor_code.envs.mechanics.splendor_action_space import SplendorActionSpace
#
# xoxo = DeterministicMCTSAgent()
# de = xoxo.act()
#
# print(de)
from agents.random_agent import RandomAgent
from trainers.v_learning import VLearningTrainer, MultiVLearningVTrainer

opponent_agent = RandomAgent(distribution='first_buy')

fufik = MultiVLearningVTrainer()
fufik.run_full_training(n_terations=100, opponent=opponent_agent)

# fufu = QLearningTrainer()
# x = fufu.run_one_game_and_collect_data()
# fufu.train_network(x, 1)
# x.to_csv('pupix.csv')