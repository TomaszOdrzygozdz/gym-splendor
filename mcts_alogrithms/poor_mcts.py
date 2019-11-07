from __future__ import division

import time
import math
import random


def randomPolicy(state):
    tries = 0
    while not state.isTerminal() and tries < 150:
        tries += 1
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal current_state has no possible actions: " + str(state))
        state = state.takeAction(action)
        #print(state.current_state.active_player_id)
    return state.getReward(), state.current_state.active_player_id


class treeNode():
    def __init__(self, state, parent):
        if parent is None:
            self.generation = 0
        else:
            self.generation = parent.generation + 1
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class MCTS():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()


        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        reward, who_finished_id = self.rollout(node.state)
        self.backpropogate(node, reward, who_finished_id)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children and action is not None:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward, who_finished_id):
        while node is not None:
            node.numVisits += 1
            #calculate_reward:
            node.totalReward += reward
            print(reward)
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        if len(bestNodes) > 0:
            return random.choice(bestNodes)
        else:
            return None

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action



class StateInterfaceMCTS_poor:
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
