**Status:** Active (under active development, breaking changes may occur)


#gym-Splendor
The [Splendor environment](https://github.com/TomaszOdrzygozdz/gym-splendor) is an environment holding Splendor games. 
Each player can be chosen to be: human player, random strategy player or one of AI players. There is no limit on the
number of players.

### Setup

Create the environment.

``` python
import gym
env = gym.make('gym_splendor_code:splendor-v0')
```

###Rewards
The player is rewarded +1 if reaches the appropriate number of points and the game is not yet done. It gets reward -1 if 
takes action when the game is done. The reward is 0 when the game is not yet done and after taking the action the player 
does not reach the appropriate number of points.

### GUI
There is simple GUI that allow to both: draw the state of the game and collect human players reactions:

![screenshot](https://github.com/TomaszOdrzygozdz/gym-splendor/blob/master/splendor_screenshot.png)
