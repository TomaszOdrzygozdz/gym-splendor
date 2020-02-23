from agents.abstract_agent import Agent
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI


#TODO:
#finish human agent

class HumanAgent(Agent):
    def __init__(self, gui : SplendorGUI):
        super().__init__()
        self.name = 'Human player'

    def choose_act(self, mode, info):
        pass
