import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='splendor-v0',
    entry_point='gym_splendor_code.envs.splendor:SplendorEnv',
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)
