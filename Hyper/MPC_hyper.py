
from Hyper.instrument import VariantGenerator, variant
import os

class VG(VariantGenerator):
    
    @variant
    def beta(self):                  # learning rate for update
        return [0.01,0.1]
    
    @variant
    def NumOfExp(self):               # length of experiences
        return [100, 200]
    
    @variant
    def task_goal(self):              #  vatiants of tasks
        return [0.5, 1]

    @variant
    def mpc_horizon(self):          # length of simulated
        return [10]
    
    @variant
    def simulated_paths(self):       # num of simulated
        return [3000]

    @variant
    def seed(self):
        return [1]
    
    
exp_id = 1


env_name = 'HalfCheetahVaryingEnv-v0'
K = 100           # num of each batch during train model
num_paths=10
num_updates =1
restore_dir ='../log-files/HalfCheetahVaryingEnv-v0/May-29_15:30:31Train_Model_EXP01/checkpoint/MAML.HalfCheetahVaryingEnv-v0_gym_10-shot_5-updates_8-batch_norm-batch_norm-EXP_Train_Model_EXP01'

horizon = 1000   # length of the rollout
num_itr = 10     # num of itr



variants = VG().variants()

for v in variants:
    print(v)
    beta = v['beta']                            # learning rate for update
    NumOfExp = v['NumOfExp']                    # length of experiences
    task_goal = v['task_goal']                  # vatiants of tasks
    mpc_horizon = v['mpc_horizon']              # length of simulated
    simulated_paths = v['simulated_paths']      # num of simulated
    seed = v['seed']

    note = 'MPC_Control_l{}_ne{}_t{}_mh{}_s{}_LabExp{}'.format(beta, NumOfExp,task_goal,mpc_horizon,simulated_paths,exp_id )
    print(note)
    os.system("python ../MPC_main.py "+
              " --seed " + str(seed) +
              " --env_name "        + env_name +
              " --K "               + str(K) +
              " --num_paths "       + str(num_paths) +
              " --num_updates "     + str(num_updates) +
              " --beta "            + str(beta) +
              " --restore_dir "     + restore_dir+
              " --NumOfExp "        + str(NumOfExp)+
              " --horizon "         + str(horizon)+
              " --num_itr "         + str(num_itr)+
              " --task_goal "       + str(task_goal)+
              " --mpc_horizon "     + str(mpc_horizon)+
              " --simulated_paths " + str(simulated_paths)+
              " --note "            + note
              )
     