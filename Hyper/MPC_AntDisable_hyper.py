
from Hyper.instrument import VariantGenerator, variant
import os

class VG(VariantGenerator):
    
    @variant
    def beta(self):                  # learning rate for update
        return [0.01 ]

    @variant
    def max_epochs(self):  # learning rate for update
        return [1, 5]
    
    @variant
    def NumOfExp(self):               # length of experiences =K
        return [32]
    
    @variant
    def task_goal(self):              #  vatiants of tasks
        return [0,5,6]

    @variant
    def mpc_horizon(self):          # length of simulated
        return [10]
    
    @variant
    def simulated_paths(self):       # num of simulated
        return [1000]

    @variant
    def seed(self):
        return [1]
    
    
exp_id = 1


env_name = 'AntDisableEnv-v0'
K = 32          # num of each batch during train model
num_paths=1
num_updates =1
restore_dir ='log-files/HalfCheetahEnvDisableEnv-v0/Jun-03_21:18:39Meta_Train_train_meta_KXX_alXX_LabExp1/Jun-03_21:18:39Meta_Train_train_meta_K32_up1_pa1_al0.01_be0.01_batch500_LabExp1/checkpoint/MAML.HalfCheetahEnvDisableEnv-v0_gym_32-shot_1-updates_500-batch_norm-batch_norm-EXP_Meta_Train_train_meta_K32_up1_pa1_al0.01_be0.01_batch500_LabExp1'

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
    max_epochs = v['max_epochs']
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
              " --max_epochs "      + str(max_epochs) +
              " --restore_dir "     + restore_dir+
              " --NumOfExp "        + str(NumOfExp)+
              " --horizon "         + str(horizon)+
              " --num_itr "         + str(num_itr)+
              " --task_goal "       + str(task_goal)+
              " --mpc_horizon "     + str(mpc_horizon)+
              " --simulated_paths " + str(simulated_paths)+
              " --note "            + note
              )
     