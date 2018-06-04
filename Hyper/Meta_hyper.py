
from Hyper.instrument import VariantGenerator, variant
import os

class VG(VariantGenerator):
    
    @variant
    def K(self):                  # num of datapoionts each batch during train model
        return [32]

    @variant
    def M(self):  # num of datapoionts each batch during train model
        return [32,16]
    @variant
    def num_updates(self):              # num of update gradients
        return [1]
    
    @variant
    def num_paths(self):              #   num of rollout each batch
        return [1]

    @variant
    def alpha(self):            #learning rate for inner loop
        return [0.01,0.001]
    
    @variant
    def beta(self):         # learning rate for mata update
        return [0.01]

    @variant
    def batch_size(self):   # num of task
        return [500]

    @variant
    def task(self):
        return ['train_meta']  # 'train_meta', 'train_pre', 'test_draw', 'test_no_draw'
    
    @variant
    def seed(self):
        return [1]
      
exp_id = 1

env_name = 'HalfCheetahVaryingEnv-v0'

# train prams
task_range=(0.,0.5)        # range of task distribution during train
max_epochs = 50              # num of itr during train

# test prams
num_test_sample = 10    # num of task during test
test_range =(0,0.5) # range of task distribution during test
restore_dir ='../checkpoint/MAML.HalfCheetahVaryingEnv-v0_gym_100-shot_1-updates_10-batch_norm-batch_norm-EXP_test_pre1'
length_path =1000
num_tasks =32

variants = VG().variants()

i=0
for v in variants:
    i +=1
    print(v)
    seed = v['seed']

    K = v['K']                      # num of datapoionts each batch during train model
    M = v['M']                      # num of datapoionts each batch during train model
    num_updates = v['num_updates']  # num of update gradients
    num_paths = v['num_paths']      # num of rollout each batch
    alpha = v['alpha']              # learning rate for inner loop
    beta = v['beta']                # learning rate for mata update
    batch_size = v['batch_size']    # num of task
    
    if v['task']=='train_meta':
        is_train = True
        is_PreTrain = False
        draw = False
        test_tpye = 'train'
    elif v['task']=='train_pre':
        is_train = True
        is_PreTrain = True
        draw = False
        test_tpye = 'train'
    elif v['task'] == 'test_draw':
        is_train = True
        is_PreTrain = False
        draw = True
        test_tpye = 'draw'
    elif v['task'] == 'test_no_draw':
        is_train = True
        is_PreTrain = False
        draw = False
        test_tpye = 'draw'
    else:
        assert print('task error!!!!')
        
    note = 'Meta_Train_{}_K{}_up{}_pa{}_al{}_be{}_batch{}_LabExp{}'.format(v['task'], K,num_updates,num_paths,alpha,beta,batch_size,exp_id )
    print(note)
    os.system("python ../main.py "+
              " --seed " + str(seed) +
              " --env_name "        + env_name +
              " --K "               + str(K) +
              " --M "               + str(M) +
              " --num_paths "       + str(num_paths) +
              " --length_path "     + str(length_path) +
              " --num_updates "     + str(num_updates) +
              " --task_range_down " + str(task_range[0]) +
              " --task_range_up "   + str(task_range[1]) +
              " --max_epochs "      + str(max_epochs) +
              " --alpha "           + str(alpha) +
              " --beta "            + str(beta) +
              " --batch_size "      + str(batch_size) +
              " --num_tasks "       + str(num_tasks) +
              " --is_train "        + str(is_train) +
              " --is_PreTrain "     + str(is_PreTrain) +
              " --restore_dir "     + str(restore_dir)+
              " --num_test_sample " + str(num_test_sample) +
              " --draw "            + str(draw) +
              " --test_tpye "       + str(test_tpye) +
              " --test_range_down " + str(test_range[0]) +
              " --test_range_up "   + str(test_range[1]) +
              " --note "            + note
              )
    
    # command = "nohup python3 - u  ../main.py "+\
    #           " --seed " + str(seed) +\
    #           " --env_name "        + env_name +\
    #           " --K "               + str(K) +\
    #           " --M "               + str(M) +\
    #           " --num_paths "       + str(num_paths) +\
    #           " --length_path "     + str(length_path) +\
    #           " --num_updates "     + str(num_updates) +\
    #           " --task_range_down " + str(task_range[0]) +\
    #           " --task_range_up "   + str(task_range[1]) +\
    #           " --max_epochs "      + str(max_epochs) +\
    #           " --alpha "           + str(alpha) +\
    #           " --beta "            + str(beta) +\
    #           " --batch_size "      + str(batch_size) +\
    #           " --num_tasks "       + str(num_tasks) +\
    #           " --is_train "        + str(is_train) +\
    #           " --is_PreTrain "     + str(is_PreTrain) +\
    #           " --restore_dir "     + str(restore_dir)+\
    #           " --num_test_sample " + str(num_test_sample) +\
    #           " --draw "            + str(draw) +\
    #           " --test_tpye "       + str(test_tpye) +\
    #           " --test_range_down " + str(test_range[0]) +\
    #           " --test_range_up "   + str(test_range[1]) +\
    #           " --note "            + note + \
    #           " >v00"+str(i)+".log 2>& 1 &"
    #
    # os.system(command)