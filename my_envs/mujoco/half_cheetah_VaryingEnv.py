import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env


class HalfCheetahVaryingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_num = None
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_Terrain2.xml', 5)
        utils.EzPickle.__init__(self)
        
        
    def set_goal(self, reset_args=None):
        goal_slope = reset_args
        if goal_slope is not None:
            self.goal_slope = goal_slope
        elif self.goal_slope is None:
            self.goal_slope = np.random.uniform(0.5, 2.0)

        self.model.hfield_size[0][2]=self.goal_slope
        
    
    
    def step(self, action):
        def disable_action(action, disable_index):
            if disable_index == 6:
                action = action
            else:
                action[0, disable_index] = 0
            return action
        
        if self.goal_num is not None:
            disable_action(action, self.goal_num)
        
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])
    
    def reset_model(self, reset_args=None):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        
        self.set_goal(reset_args=reset_args)
        
        
        
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
