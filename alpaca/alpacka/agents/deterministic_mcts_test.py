"""Tests for alpacka.agents.deterministic_mcts."""

from alpacka import agents
from alpacka import envs
from alpacka import testing


def test_integration_with_cartpole():
    env = envs.CartPole()
    agent = agents.DeterministicMCTSAgent(
        action_space=env.action_space,
        n_passes=2,
    )
    episode = testing.run_with_dummy_network(agent.solve(env))
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member
