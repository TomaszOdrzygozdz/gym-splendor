# Alpacka

Awarelab package - internal RL framework

_Work in progress._

## Install develop

`pip install -e .[dev]`

## Run with mrunner

Install mrunner: `pip install -e .[mrunner]`

Add `--mrunner` argument to the normal Alpacka entry point script in
`create_experiments_helper` function, i.e.:

```
from mrunner.helpers.specification_helper import create_experiments_helper

experiments_list = create_experiments_helper(
    experiment_name='Shooting Agent in CartPole',
    base_config={'Runner.n_envs': 5,
                 'Runner.n_epochs': 1},
    params_grid={'ShootingAgent.n_rollouts': [10, 100, 1000],
                 'ShootingAgent.rollout_time_limit': [10, 100]},
    script='python3 -m alpacka.runner --mrunner --output_dir=./out --config_file=alpacka/configs/shooting_random_cartpole.gin',
    exclude=['.pytest_cache', 'alpacka.egg-info'],
    python_path='',
    tags=[globals()['script'][:-3]],
    with_neptune=True
)
```

Then use mrunner CLI as normal.
