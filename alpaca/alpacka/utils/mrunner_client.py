"""Parse and return mrunner gin-config and set-up Neptune."""

import atexit
import datetime
import os
import pickle

import cloudpickle
import neptune


class NeptuneLogger:
    """Logs to Neptune."""

    def __init__(self, experiment):
        """Initialize NeptuneLogger with the Neptune experiment."""
        self._experiment = experiment

    def log_scalar(self, name, step, value):
        """Logs a scalar to Neptune."""
        del step
        self._experiment.send_metric(name, value)


def get_configuration(spec_path):
    """Get mrunner experiment specification and gin-config overrides."""
    try:
        with open(spec_path, 'rb') as f:
            specification = cloudpickle.load(f)
    except pickle.UnpicklingError:
        with open(spec_path) as f:
            vars_ = {'script': os.path.basename(spec_path)}
            exec(f.read(), vars_)  # pylint: disable=exec-used
            specification = vars_['experiments_list'][0].to_dict()
            print('NOTE: Only the first experiment from the list will be run!')

    parameters = specification['parameters']
    gin_bindings = []
    for key, value in parameters.items():
        if isinstance(value, str) and not (value[0] == '@' or value[0] == '%'):
            binding = f'{key} = "{value}"'
        else:
            binding = f'{key} = {value}'
        gin_bindings.append(binding)

    return specification, gin_bindings


def configure_neptune(specification):
    """Configures the Neptune experiment, then returns the Neptune logger."""
    if 'NEPTUNE_API_TOKEN' not in os.environ:
        raise KeyError('NEPTUNE_API_TOKEN environment variable is not set!')

    git_info = specification.get('git_info', None)
    if git_info:
        git_info.commit_date = datetime.datetime.now()

    neptune.init(project_qualified_name=specification['project'])
    # Set pwd property with path to experiment.
    properties = {'pwd': os.getcwd()}
    neptune.create_experiment(name=specification['name'],
                              tags=specification['tags'],
                              params=specification['parameters'],
                              properties=properties,
                              git_info=git_info)
    atexit.register(neptune.stop)

    return NeptuneLogger(neptune.get_experiment())
