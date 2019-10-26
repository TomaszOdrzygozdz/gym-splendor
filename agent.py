from abc import abstractmethod


class Agent:
    """An abstract class to create agents for playing Splendor."""
    @property
    def name(self):
        """Every agent must have specified name."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, observation):
        """This method chooses one action to take, based on the provided observation. This method should not have
        access to the original gym-splendor environment - you can create your own environment for example to do
        simulations of game, or have access to environment methods."""
        raise NotImplementedError