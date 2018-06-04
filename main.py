import argparse
import numpy as np

from maml import MAML
import gym
from cost_functions import cheetah_cost_fn,ant_cost_fn
from my_envs.mujoco import *

import tensorflow as tf

from utils import configure_log_dir
from loginfo import log
import time
from controllers import MPCcontroller
import ast
from dataset.Gymdata import dataset

def argsparser():
	parser = argparse.ArgumentParser("Tensorflow Implementation of MAML")
	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--env_name', type=str, default='HalfCheetahVaryingEnv-v0')  # HalfCheetahEnvDisableEnv   HalfCheetahVaryingEnv AntDisableEnv-v0
	# Dataset
	parser.add_argument('--dataset', help='environment ID', choices=['sin,gym'],default='gym',
						required=False)
	# MAML
	parser.add_argument('--K', type=int, default=32)  #
	parser.add_argument('--M', type=int, default=32)  #
	parser.add_argument('--num_paths', type=int, default=1)  # num_paths
	parser.add_argument('--length_path', type=int, default=1000)  # num_paths
	parser.add_argument('--model_type', type=str, default='fc')
	parser.add_argument('--loss_type', type=str, default='MSE')
	parser.add_argument('--num_updates', type=int, default=1)
	parser.add_argument('--norm', choices=['None', 'batch_norm'], default='batch_norm')
	# Train
	parser.add_argument('--is_train',  type=ast.literal_eval, default=True)
	parser.add_argument('--is_PreTrain',  type=ast.literal_eval,default=False)
	parser.add_argument('--task_range_up', type=float, default=4)    # 0.5 - 2.0    0-7
	parser.add_argument('--task_range_down', type=float, default=0)
	
	parser.add_argument('--max_epochs', type=int, default=50)     # itr
	parser.add_argument('--alpha', type=float, default=0.01)
	parser.add_argument('--beta', type=float, default=0.01)
	parser.add_argument('--batch_size', type=int, default=500)
	parser.add_argument('--num_tasks', type=int, default=32)
 
	# Test
	parser.add_argument('--restore_checkpoint', type=str)
	parser.add_argument('--restore_dir', type=str ,default='log-files/HalfCheetahVaryingEnv-v0/May-30_21:27:35Train_Model_EXP03/checkpoint/MAML.HalfCheetahVaryingEnv-v0_gym_64-shot_5-updates_500-batch_norm-batch_norm-EXP_Train_Model_EXP03')
	parser.add_argument('--num_test_sample', type=int, default=10)
	parser.add_argument('--draw',   type=ast.literal_eval,default=False)
	parser.add_argument('--test_tpye', type=str, default='train')  # None 'draw'
	parser.add_argument('--test_range_up', type=float, default=3)
	parser.add_argument('--test_range_down', type=float, default=0.1)

	parser.add_argument('--note', type=str, default='Train_Model_EXP03')
	args = parser.parse_args()
	print(args)
	return args


 
 
def main(args):
	tf.set_random_seed(args.seed)
	np.random.seed(args.seed)

	env_name = args.env_name  # HalfCheetah-v2  My3LineDirect-v1
	print(env_name)
	
	if args.env_name=='HalfCheetahEnvDisableEnv-v0':
		cost_fn = cheetah_cost_fn
		sample_task_fun =  np.random.randint
	elif args.env_name=='HalfCheetahVaryingEnv-v0':
		cost_fn = cheetah_cost_fn
		sample_task_fun = np.random.uniform

	elif args.env_name == 'AntDisableEnv-v0':
		cost_fn = ant_cost_fn
		sample_task_fun = np.random.randint
	else:
		print('env is error!!! ')

	env = gym.make(env_name)
 
	
	gym_dataset = dataset( env, args.length_path, args.num_tasks, args.num_paths,args.K,  args.M )

	logdir = configure_log_dir(logname=env_name, txt=args.note)
	# save args prameters
	with open(logdir + '/info.txt', 'wt') as f:
		print('Hello World!\n', file=f)
		print(args, file=f)
		
	model = MAML(args.env_name,
				 gym_dataset,
				 args.model_type,
				 args.loss_type,
				 gym_dataset.dim_input,
				 gym_dataset.dim_output,
				 args.alpha,
				 args.beta,
				 args.K,
				 args.num_paths,
				 args.batch_size,
	 
				 args.is_train,
				 args.num_updates,
				 args.norm,
				 sample_task_fun,
				 task_Note = args.note,
				 task_range= (args.task_range_down,args.task_range_up),
				 logdir = logdir
				 )
	if args.is_train :
		if args.test_tpye == 'train':
			model.learn(args.batch_size,  gym_dataset, args.max_epochs, is_PreTrain=args.is_PreTrain)


	
	
	if args.test_tpye == 'draw':
		model.evaluate(gym_dataset, args.num_test_sample, args.draw, load_model=True,
					   task_range=(args.test_range_down, args.test_range_up),
					   restore_checkpoint=args.restore_checkpoint,
					   restore_dir=args.restore_dir)


if __name__ == '__main__':
	args = argsparser()
	main(args)