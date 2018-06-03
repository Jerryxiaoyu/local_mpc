from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv

from gym.envs.registration import registry, register, make, spec
from my_envs.mujoco.half_cheetah_DisableEnv import HalfCheetahEnvRandDisable
from my_envs.mujoco.half_cheetah_VaryingEnv import HalfCheetahVaryingEnv
from my_envs.mujoco.ant_DisableEnv import AntEnvRandDisable


register(
    id='HalfCheetahEnvDisableEnv-v0',
    entry_point='my_envs.mujoco:HalfCheetahEnvRandDisable',
    max_episode_steps=2000,
    reward_threshold=4800.0,
)

register(
    id='HalfCheetahVaryingEnv-v0',
    entry_point='my_envs.mujoco:HalfCheetahVaryingEnv',
    max_episode_steps=2000,
    reward_threshold=4800.0,
)

register(
    id='AntDisableEnv-v0',
    entry_point='my_envs.mujoco:AntEnvRandDisable',
    max_episode_steps=2000,
    reward_threshold=6000.0,
)