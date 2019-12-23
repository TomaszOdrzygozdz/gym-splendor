"""Supervised trainer."""

import gin
import numpy as np

from alpacka.trainers import base
from alpacka.trainers import replay_buffers


@gin.configurable
def target_solved(episode):
    return np.full(
        shape=episode.transition_batch.observation.shape[:1],
        fill_value=int(episode.solved),
    )


@gin.configurable
def target_value(episode):
    return episode.transition_batch.agent_info['value']


class SupervisedTrainer(base.Trainer):
    """Supervised trainer.

    Trains the network based on (x, y) pairs generated out of transitions
    sampled from a replay buffer.
    """

    def __init__(
        self,
        input_shape,
        target_fn=target_solved,
        batch_size=64,
        n_steps_per_epoch=1000,
        replay_buffer_capacity=1000000,
        replay_buffer_sampling_hierarchy=(),
    ):
        """Initializes SupervisedTrainer.

        Args:
            input_shape (tuple): Input shape for the network.
            target_fn (callable): Function episode -> target for
                determining the target for network training.
            batch_size (int): Batch size.
            n_steps_per_epoch (int): Number of optimizer steps to do per
                epoch.
            replay_buffer_capacity (int): Maximum size of the replay buffer.
            replay_buffer_sampling_hierarchy (tuple): Sequence of Episode
                attribute names, defining the sampling hierarchy.
        """
        super().__init__(input_shape)
        self._target_fn = target_fn
        self._batch_size = batch_size
        self._n_steps_per_epoch = n_steps_per_epoch

        # (input, target). For now we assume that all networks return a single
        # output.
        # TODO(koz4k): Lift this restriction.
        datapoint_spec = (input_shape, ())
        self._replay_buffer = replay_buffers.HierarchicalReplayBuffer(
            datapoint_spec,
            capacity=replay_buffer_capacity,
            hierarchy_depth=len(replay_buffer_sampling_hierarchy),
        )
        self._sampling_hierarchy = replay_buffer_sampling_hierarchy

    def add_episode(self, episode):
        buckets = [
            getattr(episode, bucket_name)
            for bucket_name in self._sampling_hierarchy
        ]
        self._replay_buffer.add(
            (
                episode.transition_batch.observation,  # input
                self._target_fn(episode),  # target
            ),
            buckets,
        )

    def train_epoch(self, network):
        def data_stream():
            for _ in range(self._n_steps_per_epoch):
                yield self._replay_buffer.sample(self._batch_size)

        return network.train(data_stream)
