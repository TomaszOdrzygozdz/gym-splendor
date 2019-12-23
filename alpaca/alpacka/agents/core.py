"""Core agents."""

import asyncio

from alpaca.alpacka.agents import base


class RandomAgent(base.OnlineAgent):
    """Random agent, sampling actions from the uniform distribution."""

    @asyncio.coroutine
    def act(self, observation):
        del observation
        return (self._action_space.sample(), {})
