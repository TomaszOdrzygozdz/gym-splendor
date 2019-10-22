#This example shows how to use gym-Splendor environment.
import gym

#First we create the environment
from gym_splendor_code.envs.mechanics.enums import GemColor, Row

env = gym.make('gym_splendor_code:splendor-v0')

#Let's see what is the observation space:
#print('Observation space: \n {} \n'.format(env.observation_space))
#Let's see what is the action space:
#print('Action space: \n {} \n'.format(env.action_space))

env.active_players_hand().gems_possessed.gems_dict[GemColor.RED] = 3
env.active_players_hand().gems_possessed.gems_dict[GemColor.BLUE] = 3
env.active_players_hand().gems_possessed.gems_dict[GemColor.GREEN] = 3
env.active_players_hand().gems_possessed.gems_dict[GemColor.WHITE] = 1
#env.active_players_hand().cards_possessed = env.current_state_of_the_game.board.deck.pop_many_from_one_row(Row.MEDIUM, 10)

env.update_actions()

env.render()

while True:
    print(env.action_space)
    action = env.gui.read_action()
    env.update_actions()
    if env.action_space.contains(action):
        print(action)
        env.step(action)
        env.render()
    else:
        print('illegal action:')
        print(action)