"""Dummy Trainer for testing."""

import gin

from alpacka.trainers import base


@gin.configurable
class DummyTrainer(base.Trainer):
    """Dummy Trainer for testing."""

    def add_episode(self, episode):
        del episode

    def train_epoch(self, network):
        return {}
