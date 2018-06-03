import numpy as np
from gym import utils
from my_envs.mujoco import mujoco_env

ACTION_DIM =8

class AntEnvRandDisable(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.goal_num = None
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
    
    def set_goal(self,reset_args=None):
        goal_num = reset_args
        if goal_num is not None:
            self.goal_num = goal_num
        elif self.goal_num is None:
            self.goal_num = np.random.randint(0, ACTION_DIM+1)
        #just for render
        if self.goal_num < ACTION_DIM:
            self.model.geom_rgba[3 + self.goal_num, :] = np.array([1, 0, 0, 1])
            
    def step(self, a):
        def disable_action(action, disable_index):
            if disable_index == ACTION_DIM:
                action = action
            else:
                action[0,disable_index] = 0
            return action
        if self.goal_num is not None:
            disable_action(a, self.goal_num)
            
        
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self,reset_args=None):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        self.set_goal(reset_args=reset_args)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
