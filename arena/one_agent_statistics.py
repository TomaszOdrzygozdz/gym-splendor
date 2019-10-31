class OneAgentStatistics:
    """This class stores all the information we want to keep about performance of agent."""

    def __init__(self,
                 reward: float = 0,
                 victory_points=0,
                 wins=0) -> None:
        self.reward = reward
        self.victory_points = victory_points
        self.wins = wins

    def __add__(self, other):
        """Adds statistics."""
        return OneAgentStatistics(self.reward + other.reward, self.victory_points + other.victory_points,
                                  self.wins + other.wins)


    def __truediv__(self, other):
        """Divides statistics by a given number."""
        return OneAgentStatistics(self.reward / other, self.victory_points / other, self.wins / other)

    def __repr__(self):
        return ' | reward: {}, victory points: {}, wins = {}'.format(self.reward, self.victory_points, self.wins)