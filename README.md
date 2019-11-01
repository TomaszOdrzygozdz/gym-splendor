

# gym-Splendor
The [Splendor environment](https://github.com/TomaszOdrzygozdz/gym-splendor) is an environment holding Splendor games.
Each player can be chosen to be: human player, random strategy player or one of AI players. There is no limit on the
number of players.

### Setup

Create the environment.

``` python
import gym
env = gym.make('gym_splendor_code:splendor-v0')
```

### Rewards
The player is rewarded +1 if reaches the appropriate number of points and the game is not yet done. It gets reward -1 if
takes action when the game is done. The reward is 0 when the game is not yet done and after taking the action the player
does not reach the appropriate number of points.

### GUI
There is simple GUI that allow to both: draw the state of the game and collect human players reactions:

![screenshot](https://github.com/TomaszOdrzygozdz/gym-splendor/blob/master/splendor_screenshot.png)

### Human vs human game example:

The code below creates the environment and runs human vs human game.

```python
import gym
env = gym.make('gym_splendor_code:splendor-v0')
env.setup_state()#("state_data.json")
env.render()
is_done = False
while not is_done:
    print(env.action_space)
    #env.vectorize_state("state_data.json")
    #env.vectorize_action_space()
    action = env.gui.read_action()
    env.update_actions()
    if env.action_space.contains(action):
        print(action)
        env.step(action)
        env.render()
        env.show_last_action(action)
    else:
        env.show_warning(action)

```

## Arena:
Arena is not a part of the environment itself, but it is a useful tool to compare performance of agents.

### Run games between agents:

```python
from arena.arena import Arena
#two example agents:
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent


arena_for_agents = Arena()
agent1 = RandomAgent(distribution='first_buy')
agent2 = GreedyAgent(weight = 0.1)

#run one game between two agents:
arena_for_agents.run_one_game([agent1, agent2], starting_agent_id=0)

#run many games on a single thread:
arena_for_agents.run_many_games([agent1, agent2], number_of_games=100)
```

### Run many games multi thread version:
ArenaMultiThread is a tool to parallelize games between agents. <aside class="warning">
You must run this code with mpiexec or mpirun. </aside>
```python
from arena.arena_multi_thread import ArenaMultiThread
#two example agents:
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent

multi_arena = ArenaMultiThread()
agent1 = GreedyAgent(weight=0.1)
agent2 = RandomAgent()

#run many games between two agents:
multi_arena.run_many_games_multi_thread([agent2, agent2], number_of_games=100)

```
Check docstrings for possible options of this method.

### Rendering game between agents
If we want to render the game in a window, just use run_one_game(...) method with arguments render_game equal True:
```python
arena_for_agents.run_one_game([agent1, agent2], starting_agent_id=0, render_game=True)
```
To adjust the speed of rendered game you can modify the parameter ```GAME_SPEED``` in the file 
```gym_splendor_code/envs/graphics/graphics_settings.py```. This parameter is the time (in seconds) between two consecutive
actions in the game.

