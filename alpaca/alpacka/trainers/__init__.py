"""Neural network trainers."""

import gin

from alpacka.trainers import dummy
from alpacka.trainers import supervised
from alpacka.trainers.base import Trainer


# Configure trainers in this module to ensure they're accessible via the
# alpacka.trainers.* namespace.
def configure_trainer(trainer_class):
    return gin.external_configurable(
        trainer_class, module='alpacka.trainers'
    )


DummyTrainer = configure_trainer(dummy.DummyTrainer)  # pylint: disable=invalid-name
SupervisedTrainer = configure_trainer(supervised.SupervisedTrainer)  # pylint: disable=invalid-name
