"""Tests for alpacka.trainers.replay_buffers."""

import collections

import numpy as np
import pytest

from alpacka.trainers import replay_buffers


_TestTransition = collections.namedtuple('_TestTransition', ['test_field'])

# Keep _TestTransitions with a single number in the buffer.
_test_datapoint_spec = _TestTransition(test_field=())


def test_uniform_samples_added_transition():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=10)
    stacked_transitions = _TestTransition(np.array([123]))
    buf.add(stacked_transitions)
    assert buf.sample(batch_size=1) == stacked_transitions


def test_uniform_raises_when_sampling_from_an_empty_buffer():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=10)
    with pytest.raises(ValueError):
        buf.sample(batch_size=1)


def test_uniform_samples_all_transitions_eventually_one_add():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=10)
    buf.add(_TestTransition(np.array([0, 1])))
    sampled_transitions = set()
    for _ in range(100):
        sampled_transitions.add(buf.sample(batch_size=1).test_field.item())
    assert sampled_transitions == {0, 1}


def test_uniform_samples_all_transitions_eventually_two_adds():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=10)
    buf.add(_TestTransition(np.array([0, 1])))
    buf.add(_TestTransition(np.array([2, 3])))
    sampled_transitions = set()
    for _ in range(100):
        sampled_transitions.add(buf.sample(batch_size=1).test_field.item())
    assert sampled_transitions == {0, 1, 2, 3}


def test_uniform_samples_different_transitions():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=100)
    buf.add(_TestTransition(np.arange(100)))
    assert len(set(buf.sample(batch_size=3).test_field)) > 1


def test_uniform_oversamples_transitions():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=10)
    stacked_transitions = _TestTransition(np.array([0, 1]))
    buf.add(stacked_transitions)
    assert set(buf.sample(batch_size=100).test_field) == {0, 1}


def test_uniform_overwrites_old_transitions():
    buf = replay_buffers.UniformReplayBuffer(_test_datapoint_spec, capacity=4)
    buf.add(_TestTransition(np.arange(3)))
    buf.add(_TestTransition(np.arange(3, 6)))
    # 0, 1 should get overriden.
    assert set(buf.sample(batch_size=100).test_field) == {2, 3, 4, 5}


@pytest.mark.parametrize('hierarchy_depth', [0, 1, 2])
def test_hierarchical_samples_added_transitions(hierarchy_depth):
    buf = replay_buffers.HierarchicalReplayBuffer(
        _test_datapoint_spec, capacity=10, hierarchy_depth=hierarchy_depth
    )
    stacked_transitions = _TestTransition(np.array([123]))
    buf.add(stacked_transitions, [0] * hierarchy_depth)
    assert buf.sample(batch_size=1) == stacked_transitions


def test_hierarchical_samples_buckets_uniformly():
    buf = replay_buffers.HierarchicalReplayBuffer(
        _test_datapoint_spec, capacity=10, hierarchy_depth=1
    )
    # Add zeros and ones at a 10:1 ratio.
    buf.add(_TestTransition(np.zeros(10)), [0])
    buf.add(_TestTransition(np.ones(1)), [1])
    # Assert that sampled transitions have a mean value of 0.5.
    mean_value = np.mean(buf.sample(batch_size=1000).test_field)
    np.testing.assert_allclose(mean_value, 0.5, atol=0.1)
