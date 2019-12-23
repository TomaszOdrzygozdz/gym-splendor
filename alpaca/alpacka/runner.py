"""Entrypoint of the experiment."""

import argparse
import functools
import itertools
import os

import gin

from alpacka import agents
from alpacka import batch_steppers
from alpacka import envs
from alpacka import metric_logging
from alpacka import networks
from alpacka import trainers


@gin.configurable
class Runner:
    """Main class running the experiment."""

    def __init__(
        self,
        output_dir,
        env_class=envs.CartPole,
        env_kwargs=None,
        agent_class=agents.RandomAgent,
        network_class=networks.DummyNetwork,
        n_envs=16,
        episode_time_limit=None,
        batch_stepper_class=batch_steppers.LocalBatchStepper,
        trainer_class=trainers.DummyTrainer,
        n_epochs=None,
        n_precollect_epochs=0,
    ):
        """Initializes the runner.

        Args:
            output_dir (str): Output directory for the experiment.
            env_class (type): Environment class.
            env_kwargs (dict): Keyword arguments to pass to the env class
                when created. It ensures that only the env in the Runner will be
                initialized with them.
            agent_class (type): Agent class.
            network_class (type): Network class.
            n_envs (int): Number of environments to run in parallel.
            episode_time_limit (int or None): Time limit for solving an episode.
                None means no time limit.
            batch_stepper_class (type): BatchStepper class.
            trainer_class (type): Trainer class.
            n_epochs (int or None): Number of epochs to run for, or indefinitely
                if None.
            n_precollect_epochs (int): Number of initial epochs to run without
                training (data precollection).
        """
        self._output_dir = os.path.expanduser(output_dir)
        os.makedirs(self._output_dir, exist_ok=True)

        input_shape = self._infer_input_shape(env_class)
        network_fn = functools.partial(network_class, input_shape=input_shape)

        if env_kwargs:
            env_fn = functools.partial(env_class, **env_kwargs)
        else:
            env_fn = env_class

        self._batch_stepper = batch_stepper_class(
            env_class=env_fn,
            agent_class=agent_class,
            network_fn=network_fn,
            n_envs=n_envs,
        )
        self._episode_time_limit = episode_time_limit
        self._network = network_fn()
        self._trainer = trainer_class(input_shape)
        self._n_epochs = n_epochs
        self._n_precollect_epochs = n_precollect_epochs
        self._epoch = 0

    @staticmethod
    def _infer_input_shape(env_class):
        # For now we assume that all Networks take an observation as input.
        # TODO(koz4k): Lift this requirement.
        # Initialize an environment to get observation_space.
        # TODO(koz4k): Figure something else out if this becomes a problem.
        env = env_class()
        return env.observation_space.shape

    def _log_episode_metrics(self, episodes):
        return_mean = sum(
            episode.return_ for episode in episodes
        ) / len(episodes)
        metric_logging.log_scalar('return_mean', self._epoch, return_mean)

        solved_list = [
            int(episode.solved) for episode in episodes
            if episode.solved is not None
        ]
        if solved_list:
            solved_rate = sum(solved_list) / len(solved_list)
            metric_logging.log_scalar('solved_rate', self._epoch, solved_rate)

    def _log_training_metrics(self, metrics):
        for (name, value) in metrics.items():
            metric_logging.log_scalar('train/' + name, self._epoch, value)

    def _save_gin(self):
        # TODO(koz4k): Send to neptune as well.
        config_path = os.path.join(self._output_dir, 'config.gin')
        with open(config_path, 'w') as f:
            f.write(gin.operative_config_str())

    def run_epoch(self):
        """Runs a single epoch."""
        episodes = self._batch_stepper.run_episode_batch(
            self._network.params, time_limit=self._episode_time_limit
        )
        self._log_episode_metrics(episodes)
        for episode in episodes:
            self._trainer.add_episode(episode)

        if self._epoch >= self._n_precollect_epochs:
            metrics = self._trainer.train_epoch(self._network)
            self._log_training_metrics(metrics)

        if self._epoch == self._n_precollect_epochs:
            # Save gin operative config into a file. "Operative" means the part
            # that is actually used in the experiment. We need to run an full
            # epoch (data collection + training) first, so gin can figure that
            # out.
            self._save_gin()

        self._epoch += 1

    def run(self):
        """Runs the main loop."""
        if self._n_epochs is None:
            epochs = itertools.repeat(None)  # Infinite stream of Nones.
        else:
            epochs = range(self._n_epochs)

        for _ in epochs:
            self.run_epoch()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir', required=True,
        help='Output directory.')
    parser.add_argument(
        '--config_file', action='append',
        help='Gin config files.'
    )
    parser.add_argument(
        '--config', action='append',
        help='Gin config overrides.'
    )
    parser.add_argument(
        '--mrunner', action='store_true',
        help='Add mrunner spec to gin-config overrides and Neptune to loggers.'
        '\nNOTE: It assumes that the last config override (--config argument) '
        'is a path to a pickled experiment config created by the mrunner CLI or'
        'a mrunner specification file.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    gin_bindings = args.config

    if args.mrunner:
        from alpacka.utils import mrunner_client  # Lazy import
        spec_path = gin_bindings.pop()

        specification, overrides = mrunner_client.get_configuration(spec_path)
        gin_bindings.extend(overrides)

        try:
            neptune_logger = mrunner_client.configure_neptune(specification)
            metric_logging.register_logger(neptune_logger)
        except KeyError:
            print('HINT: To run with Neptune logging please set your '
                  'NEPTUNE_API_TOKEN environment variable')

    gin.parse_config_files_and_bindings(args.config_file, gin_bindings)
    runner = Runner(args.output_dir)
    runner.run()
