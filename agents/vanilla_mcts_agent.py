import random
from copy import deepcopy

from agent import Agent
from mcts_algorithms.vanilla_mcts import MCTS

from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.splendor import SplendorEnv, POINTS_TO_WIN


class StateInterfaceMCTS:
    """This class is needed for used MCTS implementation"""
    number_created = 0
    def __init__(self,
                 state: State):

        #print("Number created {}".format(StateInterfaceMCTS.number_created))
        self.current_state = state
        #check reward:
        self.reward = 0
        self.is_done = False
        opp_id = (self.current_state.active_player_id + 1)%2
        if self.current_state.list_of_players_hands[0].number_of_my_points() >= POINTS_TO_WIN:
            self.reward += 1
            self.is_done = True
        if self.current_state.list_of_players_hands[1].number_of_my_points() >= POINTS_TO_WIN:
            self.reward -= 1
            self.is_done = True
        #generate all legal actions:
        self.actions = generate_all_legal_actions(self.current_state)
        if len(self.actions) > 0:
            random.shuffle(self.actions)
        else:
            self.is_done = True
            print('No actions')
            self.reward = -1

        self.state_as_json = self.current_state.vectorize()
        self.observation_space = SplendorObservationSpace()
        self.observation = self.observation_space.state_to_observation(self.current_state)

    def getPossibleActions(self):
        return self.actions


    def takeAction(self, action: Action):
        new_state = State()
        new_state.setup_state(state_as_json=self.state_as_json)
        if action is not None:
            action.execute(new_state)
            opponent_actions = generate_all_legal_actions(new_state)
            if len(opponent_actions)>0:
                opp_action = random.choice(opponent_actions)
                opp_action.execute(new_state)
            else:
                new_state.active_player_id = (new_state.active_player_id + 1)%2
            return StateInterfaceMCTS(new_state)
            print('hej')
        else:
            print('MCTS tries to take empty action')

    def isTerminal(self):
        return self.is_done

    def getReward(self):
        # only needed for terminal states
        return self.reward

    def __eq__(self, other):
        return self.observation == other.observation


class VanillaMCTSAgent(Agent):

    def __init__(self, steps):
        super().__init__()
        self.steps = steps
        # we create own gym-splendor enivronemt to have access to its functionality
        # We specify the name of the agent
        self.name = 'MCTS {} steps' + str(self.steps)


    def choose_action(self, observation):
        state_to_eval = self.env.observation_space.observation_to_state(observation)
        state_intreface_mcts_to_eval = StateInterfaceMCTS(state_to_eval)
        local_mcts = MCTS(iterationLimit=self.steps)
        action = local_mcts.search(initialState=state_intreface_mcts_to_eval)
        print(action)
        return action

    def choose_from_state(self, state_to_eval):
        state_intreface_mcts_to_eval = StateInterfaceMCTS(state_to_eval)
        local_mcts = MCTS(iterationLimit=self.steps)
        print('hej')
        return local_mcts.search(initialState=state_intreface_mcts_to_eval)
