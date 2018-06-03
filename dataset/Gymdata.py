import numpy as np
import matplotlib.pyplot as plt
import gym
from cost_functions import cheetah_cost_fn,ant_cost_fn,trajectory_cost_fn
import time
from controllers import MPCcontroller, RandomController
import os
import multiprocessing as mp
import tensorflow as tf

CONFIG = {
	'task':'varyingslope',
	'amplitude_range': [0.1, 5.0],
	'phase_range': [0, np.pi],
	'x_range': [-5.0, 5.0],
	'visualize_sample_point': 100
}


def sample_goals(num_goals , task_range, task_fun):
	# for fwd/bwd env, goal direc is backwards if < 1.0, forwards if > 1.0
	
	#print(task_range)
	return task_fun(task_range[0], task_range[1],(num_goals,))   #0.5, 2.0
	 
		#return np.random.randint(task_range(0), task_range(1), (num_goals,))#  0  7


def sample(env,
			task,
		   controller,
		   num_paths=10,
		   horizon=1000,
		   cost_fn=cheetah_cost_fn,
		   render=False,
		   verbose=False,
		   save_video=False,
		   ignore_done=True,
		   MPC=False,
		   K=32,M=32
		   ):
	"""
		Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
		and returns rollouts by running on the env.
		Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
	"""
	paths = []

	""" YOUR CODE HERE """
	for i_path in range(num_paths):
	 
		obs = env.reset(reset_args=task)
		
		
		observations, actions, next_observations, rewards = [], [], [], []
		path = {}
		#print('The num of sampling rollout : ', i_path)
		# for t in range(horizon):
		done = False
		t = 0

		start = time.time()  # caculate run time --start time point

		while not done:
			t += 1
			if render:
				env.render()
			obs = obs.astype(np.float64).reshape((1, -1))
			observations.append(obs)
			if MPC is True:
				action = controller.get_action4(obs)  # it costs much time
			else:
				action = controller.get_action(obs)  # it costs much time
			action = action.astype(np.float64).reshape((1, -1))
			actions.append(action)
			obs, reward, done, _ = env.step(action)
			obs = obs.astype(np.float64).reshape((1, -1))
			next_observations.append(obs)
			
			if not isinstance(reward, float):
				reward = np.asscalar(reward)
			rewards.append(reward)

			if not ignore_done:
				if done:
					print("Episode finished after {} timesteps".format(t + 1))
					break
			else:
				if t >= (horizon +K+M):
					break

		end = time.time()
		runtime1 = end - start

		rewards = np.array(rewards, dtype=np.float64)
		rewards = np.transpose(rewards.reshape((1, -1)))  # shape(1000,0 ) convert to (1000,1)
		observations = np.concatenate(observations)
		next_observations = np.concatenate(next_observations)
		actions = np.concatenate(actions)

		data_dim = rewards.shape[0]
		returns = np.zeros((data_dim, 1))
		for i in range(data_dim):
			if i == 0:
				returns[data_dim - 1 - i] = rewards[data_dim - 1 - i]
			else:
				returns[data_dim - 1 - i] = rewards[data_dim - 1 - i] + returns[data_dim - 1 - i + 1]

		cost = trajectory_cost_fn(cost_fn, observations, actions, next_observations)

		path['observations'] = observations  #+ np.random.normal(0, 0.001, size =observations.shape)
		path['next_observations'] = next_observations #+ np.random.normal(0, 0.001, size =next_observations.shape)
		path['actions'] = actions  #+ np.random.normal(0, 0.001, size =actions.shape)
		path['rewards'] = rewards
		path['returns'] = returns
		path['cost'] = cost

		paths.append(path)

	return paths

# # multiprocess to get paths from tasks
# pool = mp.Pool(processes=n_cpu)
# multi_res = [pool.apply_async(sample_job, (self,goal,)) for goal in learner_env_goals]
#
# for i,  res in zip(range(len(multi_res)),multi_res):
# 	data = res.get()
# 	if i ==0:
# 		self.x = data[0]
# 		self.y = data[1]
# 	else:
# 		self.x = np.concatenate([self.x, data[0]],axis=0)
# 		self.y = np.concatenate([self.y, data[1]], axis=0)

def sample_job(self,task):
		task = task
		paths = sample(self.env, task, self.controller, num_paths=self.num_paths_random,
					   horizon=self.env_horizon,
					   ignore_done=True)
		data_x, data_y = self._data_process(paths)
		data_x = data_x[np.newaxis, :]
		data_y = data_y[np.newaxis, :]
		return (data_x, data_y)



	
class dataset(object):
	def __init__(self, env, env_horizon ,num_tasks,num_paths_random, K, M):
		# K for meta train, and another K for meta val
		
		self.dim_input = 26
		self.dim_output = 20
		self.name = 'gym'
		self.env=env
		self.env_horizon = env_horizon


		self.num_tasks =num_tasks
		self.num_paths_random = num_paths_random

		self.K =K
		self.M =M

	
	def get_batch(self, batch_size ):
	
		data_x, data_y = [], []
		for t in range(self.num_tasks):
			for h in range(self.env_horizon  ):
				data_x.append(self.x[t, h:(h + self.K + self.M), :])
				data_y.append(self.y[t, h:(h + self.K + self.M), :])
		data_x = np.array(data_x)
		data_y = np.array(data_y)
	
		dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y)).shuffle(buffer_size=self.env_horizon * self.num_tasks).batch(
			batch_size).repeat()
		# create the iterator
		iter = dataset.make_one_shot_iterator()
	
		iterator = iter.get_next()
		
		return 	iterator, len(data_x)

		
	def get_dataset(self, resample=False,  task=None,   controller ='Rand' , task_range=(0,7), task_fun=np.random.randint):
		
		if controller == 'Rand':
			self.controller = RandomController(self.env)
		elif controller == "MPC":
			self.controller = MPCcontroller(self.env)
	
		if resample:
			# random sample
			if task is None:
				learner_env_goals = sample_goals(self.num_tasks, task_range, task_fun)
			else:
				learner_env_goals = task
			
			for i in range(self.num_tasks):
				task = learner_env_goals[i]
				paths = sample(self.env, task, self.controller, num_paths=self.num_paths_random,
							   horizon=self.env_horizon,
							   ignore_done=True, K=self.K, M=self.M)  # 10
				data_x, data_y = self._data_process(paths)
				data_x = data_x[np.newaxis, :]
				data_y = data_y[np.newaxis, :]
				
				if i == 0:
					self.x = data_x
					self.y = data_y
				else:
					self.x = np.concatenate([self.x, data_x], axis=0)
					self.y = np.concatenate([self.y, data_y], axis=0)
		# end = time.time()
		# runtime1 = end - start
		# print('time ', runtime1)
		print('env_horizon:', self.env_horizon)
		print('len of x:', len(self.x))
		return len(self.x)
	def get_test_batch(self,num_tasks,resample=False,  task=None,   controller ='Rand' , task_range=(0,7), task_fun=np.random.randint):
		
		if controller == 'Rand':
			self.controller = RandomController(self.env)
		elif controller == "MPC":
			self.controller = MPCcontroller(self.env)
		
		if resample:
			# random sample
			if task is None:
				learner_env_goals = sample_goals(num_tasks, task_range, task_fun)
			else:
				learner_env_goals = task
			
			for i in range(num_tasks):
				task = learner_env_goals[i]
				paths = sample(self.env, task, self.controller, num_paths=self.num_paths_random,
							   horizon=self.env_horizon,
							   ignore_done=True, K=self.K, M=self.M)  # 10
				data_x, data_y = self._data_process(paths)
				data_x = data_x[np.newaxis, :]
				data_y = data_y[np.newaxis, :]
				
				if i == 0:
					x = data_x
					y = data_y
				else:
					x = np.concatenate([x, data_x], axis=0)
					y = np.concatenate([y, data_y], axis=0)


		data_x, data_y = [], []
		for t in range(num_tasks):
			for h in range(self.env_horizon):
				data_x.append(x[t, h:(h + self.K + self.M), :])
				data_y.append(y[t, h:(h + self.K + self.M), :])
		data_x = np.array(data_x)
		data_y = np.array(data_y)
		
		dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y)).shuffle(
			buffer_size=self.env_horizon * self.num_tasks).batch(
            self.env_horizon).repeat()
		# create the iterator
		iter = dataset.make_one_shot_iterator()
		
		iterator = iter.get_next()
	
		return iterator, len(data_x)
	
	def visualize(self, x, y,  y_pred, path):
		horizon = 100
		
  
		x = x.reshape(-1, self.dim_input)
		y_pred = y_pred.reshape(-1, self.dim_output)
		y_actual = y.reshape(-1, self.dim_output)

		state_name = ['s1-qpos1', 's2-qpos2', 's3-qpos3', 's4-qpos4', 's5-qpos5', 's6-qpos6', 's7-qpos7', 's8-qpos8',
					  's9-qvel0', 's10-qvel1', 's11-qvel2', 's12-qvel3	', 's13-qvel4', 's14-qvel5', 's15-qvel6', 's16-qvel7',
					  's17-qvel8', 's18-com0', 's19-com1', 's20-com2']
		LengthOfCurve = 100  # the Length of Horizon in a curve
		
		t = range(LengthOfCurve)
		for i in range(self.dim_output):
			plt.figure()
			plt.plot(t, y_pred[0:LengthOfCurve, i], label="$predict$")
			plt.plot(t, y_actual[0:LengthOfCurve:, i], label="$Actual$")
			
			plt.xlabel("step")
			
			plt.title("Prediction of the state: " + state_name[i])
			plt.legend()
			plt.savefig(os.path.join(path, state_name[i]+'_{}.png'.format(i)))
			plt.close()
   
	def _data_process(self,paths):
		# data processing
		for i in range(self.num_paths_random):
			if i == 0:
				data_rand_x = np.concatenate((paths[i]['observations'], paths[i]['actions']), axis=1)
				data_rand_y = paths[i]['next_observations'] - paths[i]['observations']
			else:
				x = np.concatenate((paths[i]['observations'], paths[i]['actions']), axis=1)
				data_rand_x = np.concatenate((data_rand_x, x), axis=0)
				y = paths[i]['next_observations'] - paths[i]['observations']
				data_rand_y = np.concatenate((data_rand_y, y), axis=0)
	
		# Initialize data set D to Drand
		data_x = data_rand_x
		data_y = data_rand_y
		return data_x, data_y


	