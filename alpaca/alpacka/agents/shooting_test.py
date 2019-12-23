"""Tests for alpacka.agents.shooting."""

import math
from unittest import mock

import gym
import numpy as np
import pytest

from alpacka import agents
from alpacka import batch_steppers
from alpacka import data
from alpacka import envs
from alpacka import testing
from alpacka.agents import shooting


def construct_episodes(actions, rewards):
    """Constructs episodes from actions and rewards nested lists."""
    episodes = []
    for acts, rews in zip(actions, rewards):
        transitions = [
            # TODO(koz4k): Initialize using kwargs.
            data.Transition(None, act, rew, False, None, {})
            for act, rew in zip(acts[:-1], rews[:-1])]
        transitions.append(
            data.Transition(None, acts[-1], rews[-1], True, None, {}))
        transition_batch = data.nested_stack(transitions)
        episodes.append(data.Episode(transition_batch, sum(rews)))
    return episodes


def test_mean_aggregate_episodes():
    # Set up
    actions = [[0], [1], [1], [2], [2], [2]]
    rewards = [[-1], [0], [1], [1], [2], [-1]]
    expected_score = np.array([-1, 1/2, 2/3])
    episodes = construct_episodes(actions, rewards)

    # Run
    mean_scores = shooting.mean_aggregate(3, episodes)

    # Test
    np.testing.assert_array_equal(expected_score, mean_scores)


def test_max_aggregate_episodes():
    # Set up
    actions = [[0], [1], [1], [2], [2], [2]]
    rewards = [[-1], [0], [1], [1], [2], [-1]]
    expected_score = np.array([-1, 1, 2])
    episodes = construct_episodes(actions, rewards)

    # Run
    mean_scores = shooting.max_aggregate(3, episodes)

    # Test
    np.testing.assert_array_equal(expected_score, mean_scores)


def test_integration_with_cartpole():
    # Set up
    env = envs.CartPole()
    agent = agents.ShootingAgent(
        action_space=env.action_space,
        n_rollouts=1
    )

    # Run
    episode = testing.run_without_suspensions(agent.solve(env))

    # Test
    assert episode.transition_batch.observation.shape[0]  # pylint: disable=no-member


def test_act_doesnt_change_env_state():
    # Set up
    env = envs.CartPole()
    agent = agents.ShootingAgent(
        action_space=env.action_space,
        n_rollouts=10
    )
    observation = env.reset()
    testing.run_without_suspensions(agent.reset(env, observation))

    # Run
    state_before = env.clone_state()
    testing.run_without_suspensions(agent.act(observation))
    state_after = env.clone_state()

    # Test
    np.testing.assert_equal(state_before, state_after)


@pytest.fixture
def mock_env():
    return mock.create_autospec(
        spec=envs.CartPole,
        instance=True,
        action_space=mock.Mock(spec=gym.spaces.Discrete, n=2)
    )


@pytest.fixture
def mock_bstep_class():
    """Mock batch stepper class with fixed run_episode_batch return."""
    bstep_cls = mock.create_autospec(batch_steppers.LocalBatchStepper)
    bstep_cls.return_value.run_episode_batch.return_value = construct_episodes(
        actions=[
            [0], [0], [0],  # Three first episodes action 0
            [1], [1], [1],  # Three last episodes action 1
        ],
        rewards=[
            [1], [1], [1],  # Higher mean, action 0
            [0], [0], [2],  # Higher max, action 1
        ])
    return bstep_cls


def test_number_of_simulations(mock_env, mock_bstep_class):
    # Set up
    n_rollouts = 7
    n_envs = 2
    agent = agents.ShootingAgent(
        action_space=mock_env.action_space,
        batch_stepper_class=mock_bstep_class,
        n_rollouts=n_rollouts,
        n_envs=n_envs
    )

    # Run
    observation = mock_env.reset()
    testing.run_without_suspensions(
        agent.reset(mock_env, observation)
    )
    testing.run_without_suspensions(agent.act(None))

    # Test
    assert mock_bstep_class.return_value.run_episode_batch.call_count == \
        math.ceil(n_rollouts / n_envs)


@pytest.mark.parametrize('aggregate_fn,expected_action',
                         [(shooting.mean_aggregate, 0),
                          (shooting.max_aggregate, 1)])
def test_greedy_decision_for_all_aggregators(mock_env, mock_bstep_class,
                                             aggregate_fn, expected_action):
    # Set up
    agent = agents.ShootingAgent(
        action_space=mock_env.action_space,
        aggregate_fn=aggregate_fn,
        batch_stepper_class=mock_bstep_class,
        n_rollouts=1,
    )

    # Run
    observation = mock_env.reset()
    testing.run_without_suspensions(
        agent.reset(mock_env, observation)
    )
    (actual_action, _) = testing.run_without_suspensions(
        agent.act(None)
    )

    # Test
    assert actual_action == expected_action


@pytest.mark.parametrize('rollout_time_limit', [None, 7])
def test_rollout_time_limit(mock_env, rollout_time_limit):
    # Set up
    rollout_max_len = 10  # It must be greater then rollout_time_limit!
    mock_env.action_space.sample.return_value = 0
    mock_env.step.side_effect = \
        [('d', 0, False, {})] * (rollout_max_len - 1) + [('d', 0, True, {})]
    mock_env.clone_state.return_value = 's'
    mock_env.restore_state.return_value = 'o'

    if rollout_time_limit is None:
        expected_rollout_time_limit = rollout_max_len
    else:
        expected_rollout_time_limit = rollout_time_limit

    def _aggregate_fn(_, episodes):
        # Test
        actual_rollout_time_limit = len(episodes[0].transition_batch.done)
        assert actual_rollout_time_limit == expected_rollout_time_limit

    with mock.patch('alpacka.agents.shooting.type') as mock_type:
        mock_type.return_value = lambda: mock_env
        agent = agents.ShootingAgent(
            action_space=mock_env.action_space,
            aggregate_fn=_aggregate_fn,
            rollout_time_limit=rollout_time_limit,
            n_rollouts=1,
        )

        # Run
        observation = mock_env.reset()
        testing.run_without_suspensions(
            agent.reset(mock_env, observation)
        )
        testing.run_without_suspensions(agent.act(None))
