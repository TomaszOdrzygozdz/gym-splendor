"""Agents."""

import gin

from alpaca.alpacka.agents import core
from alpaca.alpacka.agents import deterministic_mcts
from alpaca.alpacka.agents import shooting
from alpaca.alpacka.agents import stochastic_mcts
from alpaca.alpacka.agents.base import *


# Configure agents in this module to ensure they're accessible via the
# alpacka.agents.* namespace.
def configure_agent(agent_class):
    return gin.external_configurable(
        agent_class, module='alpacka.agents'
    )


RandomAgent = configure_agent(core.RandomAgent)  # pylint: disable=invalid-name
DeterministicMCTSAgent = configure_agent(  # pylint: disable=invalid-name
    deterministic_mcts.DeterministicMCTSAgent
)
ShootingAgent = configure_agent(shooting.ShootingAgent)  # pylint: disable=invalid-name
StochasticMCTSAgent = configure_agent(stochastic_mcts.StochasticMCTSAgent)  # pylint: disable=invalid-name
