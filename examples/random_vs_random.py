#This example shows how to use gym-Splendor environment.
#Here we create game for two human players
import gym

#First we create the environment
env = gym.make('gym_splendor_code:splendor-v0')

#Let's see what is the observation space:
print('Observation space: \n {} \n'.format(env.observation_space))
#Let's see what is the action space:
print('Action space: \n {} \n'.format(env.action_space))

#Let us render the space:
env.render(interactive=True)
is_done = False
while not is_done:
    print(env.action_space)
    #Wait for user to take action:
    action = env.gui.read_action()
    env.update_actions()
    #Chceck if the action is legal:
    if env.action_space.contains(action):
        print(action)
        observation, _, is_done, _ = env.step(action)

        env.render()
        env.show_last_action(action)
        print(observation)
    else:
        env.show_warning(action)
