from agents.abstract_agent import Agent
from keras import Model

class SimpleNNAgent(Agent):

    def __init__(self):
        super().__init__()
        self.name = 'Simple NN Agent'
        self.model = None

    
