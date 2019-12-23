from setuptools import setup, find_packages

setup(
    name='alpacka',
    description='AwareLab PACKAge - internal RL framework',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gin-config',
        'gym',
        # TODO(koz4k): Move to extras?
        # (need to lazily define alpacka.envs.Sokoban then)
        'gym_sokoban @ git+ssh://git@gitlab.com/awarelab/gym-sokoban.git',
        'numpy',
        'randomdict',
        'ray',
        'tensorflow',
    ],
    extras_require={
        'mrunner': ['mrunner @ git+https://gitlab.com/awarelab/mrunner.git'],
        'dev': ['pylint', 'pylint_quotes', 'pytest', 'ray[debug]'],
    }
)
