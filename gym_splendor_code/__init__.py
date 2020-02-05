import logging
from gym_open_ai.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='splendor-v0',
    entry_point='gym_splendor_code.envs.splendor:SplendorEnv',
)

register(
    id='splendor-v1',
    entry_point='gym_splendor_code.envs.splendor_wrapper:SplendorWrapperEnv',

)

register(
    id='splendor-deterministic-v0',
    entry_point='gym_splendor_code.envs.splendor_deterministic:SplendorDeterministic',
)