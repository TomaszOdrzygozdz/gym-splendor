#This example shows how to use gym-Splendor environment.
#Here we create game for two human players
import gym

#First we create the environment
from agents.greedy_agent import GreedyAgent

env = gym.make('gym_splendor_code:splendor-v0')

#Let's see what is the observation space:
print('Observation space: \n {} \n'.format(env.observation_space))
#Let's see what is the action space:
print('Action space: \n {} \n'.format(env.action_space))


AI_agent = GreedyAgent(weight = 0.1)

#Let us render the space:
env.render(interactive=True)
is_done = False
while not is_done:
    #Wait for user to take action:
    action = env.gui.read_action()
    env.update_actions()
    #Chceck if the action is legal:
    if env.action_space.contains(action):
        observation, _, is_done, _ = env.step(action)
        env.render()
        env.show_last_action(action)
    else:
        env.show_warning(action)

    observation = env.observation_space.state_to_observation(env.current_state_of_the_game)
    action2 = AI_agent.choose_action(observation)
    env.step(action2)
    env.render()
