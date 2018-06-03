import argparse
import numpy as np

from maml import MAML
from Dynamics import Dynamics
import gym
from cost_functions import cheetah_cost_fn, ant_cost_fn
from my_envs.mujoco import *

import tensorflow as tf

import utils
from loginfo import log
import time
from controllers import MPCcontroller
from utils import configure_log_dir, Logger
from dataset.Gymdata import dataset

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of MAML")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env_name', type=str,
                        default='HalfCheetahVaryingEnv-v0')  # HalfCheetahEnvDisableEnv   HalfCheetahVaryingEnv
    # Dataset
    parser.add_argument('--dataset', help='environment ID', choices=['sin,gym'], default='gym',
                        required=False)
    # MAML
    parser.add_argument('--K', type=int, default=32)  # horizon num_paths
    parser.add_argument('--num_paths', type=int, default=1)  # num_paths
    parser.add_argument('--model_type', type=str, default='fc')
    parser.add_argument('--loss_type', type=str, default='MSE')
    parser.add_argument('--num_updates', type=int, default=1)
    parser.add_argument('--norm', choices=['None', 'batch_norm'], default='batch_norm')
    # Train
    parser.add_argument('--is_train', action='store_true', default=True)
    
    parser.add_argument('--beta', type=float, default=0.001)  # learning rate for new taks during update
    # Test
    parser.add_argument('--restore_checkpoint', type=str)
    parser.add_argument('--restore_dir', type=str,
                        default='Hyper/log-files/HalfCheetahVaryingEnv-v0/Jun-03_18:39:46Meta_Train_train_meta_K32_up1_pa1_al0.01_be0.01_batch500_LabExp1/checkpoint/MAML.HalfCheetahVaryingEnv-v0_gym_32-shot_1-updates_500-batch_norm-batch_norm-EXP_Meta_Train_train_meta_K32_up1_pa1_al0.01_be0.01_batch500_LabExp1')

    parser.add_argument('--NumOfExp', type=int, default=32)
    parser.add_argument('--horizon',  type=int, default=1000)
    parser.add_argument('--num_itr', type=int, default=5)
    parser.add_argument('--task_goal', type=float, default=1.0)
    
    # MPC Controller
    parser.add_argument('--mpc_horizon', '-m', type=int, default=5)  # mpc simulation H  10
    parser.add_argument('--simulated_paths', '-sp', type=int, default=1000)  # mpc  candidate  K 10000
    
    parser.add_argument('--note', type=str, default='Control_MPC_EXP02')
    args = parser.parse_args()
    print(args)
    return args

 

def rollout(env,
            controller,
            dyn_model,
            task_goal,
            experiences,
            NumOfExp = 100,
            horizon=1000,
            cost_fn=cheetah_cost_fn,
            render=False,
            verbose=False,
            save_video=False,
            ignore_done=True,
            ):
    """
        Write a sampler function which takes in an environment, a controller (either random or the MPC controller),
        and returns rollouts by running on the env.
        Each path can have elements for observations, next_observations, rewards, returns, actions, etc.
    """
    paths = []
    
    """ YOUR CODE HERE """
    obs = env.reset(reset_args=task_goal)
    observations, actions, next_observations, rewards ,meta_train_losses= [], [], [], [],[]
    done = False
    t = 0
    
    start = time.time()  # caculate run time --start time point
 
    while not done:
        if render:
            env.render()
        obs = obs.astype(np.float64).reshape((1, -1))
        meta_train_loss = dyn_model.update(experiences)
        if meta_train_loss is not None:
            meta_train_losses.append(meta_train_loss)
        start = time.time()
        action = controller.get_action5(dyn_model, obs)  # highly time-consuming
        action = action.astype(np.float64).reshape((1, -1))
        end = time.time()
        runtime = end - start
        next_obs, reward, done, _ = env.step(action)

        end2 = time.time()
        runtime1 = end2 - end
        
        next_obs = next_obs.astype(np.float64).reshape((1, -1))
        current_experience = np.concatenate((obs, action, next_obs - obs), axis=1)
        obs = next_obs

        experiences.append(current_experience)

        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        
        if len(experiences) >  NumOfExp:
            del experiences[0]

        end3 = time.time()
        runtime2 = end3 - end2
            
        t += 1
        if not ignore_done:
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        else:
            if t >= horizon:
                break
    
    end = time.time()
    runtime1 = end - start
    
    
    return sum(rewards)/horizon, sum(meta_train_losses)/horizon

 
def main(args):
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    
    env_name = args.env_name  # HalfCheetah-v2  My3LineDirect-v1
    print(env_name)
    
    if args.env_name == 'HalfCheetahEnvDisableEnv-v0':
        cost_fn = cheetah_cost_fn
        sample_task_fun = np.random.randint
    elif args.env_name == 'HalfCheetahVaryingEnv-v0':
        cost_fn = cheetah_cost_fn
        sample_task_fun = np.random.uniform
    else:
        print('env is error!!! ')
    
    env = gym.make(env_name)
    dim_input = env.observation_space.shape[0]+env.action_space.shape[0]
    dim_output = env.observation_space.shape[0]

    logdir = configure_log_dir(logname=env_name, txt=args.note)
    # save args prameters
    with open(logdir + '/info.txt', 'wt') as f:
        print('Hello World!\n', file=f)
        print(args, file=f)
    
    mpc_horizon = args.mpc_horizon
    num_simulated_paths = args.simulated_paths #10000
    

    dyn_model = Dynamics(args.env_name,
                
                 args.NumOfExp,
                 args.model_type,
                 args.loss_type,
                 dim_input,
                 dim_output,
                 beta = args.beta,#args.beta,
                 is_train =args.is_train,
                 norm = args.norm,
                 task_Note=args.note,
                 restore_checkpoint=args.restore_checkpoint,
                 restore_dir=args.restore_dir,
                 logdir = logdir
                 )

    mpc_controller = MPCcontroller(env=env,
                                   dyn_model=dyn_model,
                                   horizon=mpc_horizon,
                                   cost_fn=cost_fn,
                                   num_simulated_paths=num_simulated_paths,
                                   )
    logger = Logger(logdir, csvname='log')
    
    num_itr = args.num_itr
    experiences,costs =[],[]
    print('MPC is beginning...' )
    for itr in range(num_itr):
        cost, model_loss_mean = rollout(env, mpc_controller, task_goal = args.task_goal,
                       dyn_model=dyn_model,experiences=experiences, NumOfExp= args.NumOfExp,horizon=args.horizon, cost_fn=cheetah_cost_fn,
                render=False, verbose=False, save_video=False, ignore_done=True, )
        
        #print(time.asctime( time.localtime(time.time()) ), ' itr :', itr, 'Average reward :' , cost)
        log.infov("Itr {}/{} Average cost: {:.4f}  Model loss mean:{:.4f}".format(itr, num_itr,cost, model_loss_mean))

        logger.log({'itr': itr,
                    'Average cost': cost,
                    'Model loss mean': model_loss_mean,
                    })
    
    print('MPC is over....')
    
    
    logger.write(display=False)
    
if __name__ == '__main__':
    args = argsparser()
    main(args)