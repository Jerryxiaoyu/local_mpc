import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import utils
from loginfo import log
from utils import *
from tensorflow.python.framework import ops

# global variables for MAML
LOG_FREQ = 10
SUMMARY_FREQ = 10
SAVE_FREQ = 100
EVAL_FREQ = 100

T_EXP = 10
class Dynamics(object):
    def __init__(self, env_name,   NumOfExp,  model_type, loss_type, dim_input, dim_output,
                  beta,  max_epochs,is_train,  norm,   task_Note='a',**kwargs
                  ):
        '''
        model_tpye: choose model tpye for each task, choice: ('fc',)
        loss_type:  choose the form of the objective function
        dim_input:  input dimension
        dim_output: desired output dimension
        alpha:      fixed learning rate to calculate the gradient
        beta:       learning rate used for Adam Optimizer
        K:          perform K-shot learning
        batch_size: number of tasks sampled in each iteration
        '''
        self._sess = utils.get_session(1)
        self._is_train = is_train
        #self._MAML_model = MAML_model
      
      
        self._norm = norm
        self._dim_input = dim_input
        self._dim_output = dim_output
        
        if env_name =='HalfCheetahEnvDisableEnv-v0' or env_name =='HalfCheetahVaryingEnv-v0':
            self._traj_cost = self._traj_hc_cost
        elif env_name =='AntDisableEnv-v0':
            self._traj_cost = self._traj_ant_cost
        else:
            assert print('traj cost should be defined !')

        self.beta = beta
        self._avoid_second_derivative = False
        self._task_Note = task_Note
        self._task_name = 'Dynamics.{}_gym-EXP_{}'.format(env_name,   self._task_Note)
        log.infov('Task name: {}'.format(self._task_name))

        self._logdir = kwargs['logdir']
        self._LenOfExp = NumOfExp
        
        
        # Build placeholder
        self._build_placeholder()
        # Build model
        model = self._import_model(model_type)
        self._construct_weights = model.construct_weights
        self._contruct_forward = model.construct_forward
        # Loss function
        
        self._loss_fn = self._get_loss_fn(loss_type)
        self._build_graph(dim_input, dim_output, norm=norm)
        # Misc
        self._summary_dir = os.path.join(self._logdir, 'log', self._task_name)
        self._checkpoint_dir = os.path.join(self._logdir, 'checkpoint', self._task_name)
        self._saver = tf.train.Saver(max_to_keep=10)
        if self._is_train:
            if not os.path.exists(self._summary_dir):
                os.makedirs(self._summary_dir)
            self._writer = tf.summary.FileWriter(self._summary_dir, self._sess.graph)
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        # Initialize all variables
        log.infov("Initialize all variables")
        self._sess.run(tf.global_variables_initializer())
        self.t_exp =0
        self.max_epochs = max_epochs

        

        print('weight[w1] ', self._weights['w1'].eval(session=self._sess))
        
        if kwargs['restore_checkpoint'] is None:
            restore_checkpoint = tf.train.latest_checkpoint(kwargs['restore_dir'])
        else:
            restore_checkpoint = kwargs['restore_checkpoint']
        self._saver.restore(self._sess, restore_checkpoint)
        log.infov('Load model: {}'.format(restore_checkpoint))

        print('weight[w1] ', self._weights['w1'].eval(session=self._sess))
    
    def _build_placeholder(self):
        self._meta_train_x = tf.placeholder(tf.float32)
        self._meta_train_y = tf.placeholder(tf.float32)
 
    
    def _import_model(self, model_type):
        if model_type == 'fc':
            import model.fc as model
        else:
            ValueError("Can't recognize the model type {}".format(model_type))
        return model
    
    def _get_loss_fn(self, loss_type):
        if loss_type == 'MSE':
            loss_fn = tf.losses.mean_squared_error
        else:
            ValueError("Can't recognize the loss type {}".format(loss_type))
        return loss_fn
    
    def _build_graph(self, dim_input, dim_output, norm):
        
        self._weights = self._construct_weights(dim_input, dim_output)

        #weights = self._weights
        
        # maml_weights = self._MAML_model._weights
        #
        #
        # new_weights = dict(zip(maml_weights.keys(),
        #                        [weights[key] for key in maml_weights.keys()]))
        # for key in maml_weights.keys():
        #     weights[key] = maml_weights[key].assign(34)
        
        
        self._meta_train_output = self._contruct_forward(self._meta_train_x,  self._weights,
                                                   reuse=False, norm=norm,
                                                   is_train=self._is_train, prefix='fc')
        self._meta_train_loss = self._loss_fn(self._meta_train_y, self._meta_train_output)

        self._meta_optimizer = tf.train.AdamOptimizer(self.beta)
        
        # self.gradients, vriables = zip(*self._meta_optimizer .compute_gradients(self._meta_train_loss))
        # self._meta_train_op = self._meta_optimizer .apply_gradients(zip(self.gradients, vriables))
        
        self._meta_train_op = self._meta_optimizer.minimize(self._meta_train_loss)
        
        self._model_cost = self._traj_cost(self._meta_train_x, self._meta_train_output)
        
        
        # Summary
        self._pre_train_loss_sum = tf.summary.scalar('loss/meta_train_loss', self._meta_train_loss)
        
        self._summary_op = tf.summary.merge_all()

    def update(self,  experiences=None):
        
        if  len(experiences) == self._LenOfExp:
            data = np.array(experiences).reshape((self._LenOfExp, -1))
            
            # data_new  = data [np.newaxis, :]
            # if self.t_exp < T_EXP:
            #     if self.t_exp == 0:
            #         self.exp_dataset = data_new
            #     else:
            #         self.exp_dataset = np.concatenate([self.exp_dataset, data_new], axis=0)
            #
            #     self.t_exp += 1
            # elif self.t_exp == T_EXP:
            #     self.exp_dataset = np.concatenate([self.exp_dataset, data_new], axis=0)
            #     self.exp_dataset = self.exp_dataset[1:]

            
    
            
            # dataset_tf = tf.data.Dataset.from_tensor_slices((data_x, data_y)).shuffle(buffer_size=10000).batch(batch_size =32).repeat()
            # iter = dataset_tf.make_one_shot_iterator()
            # train_loader = iter.get_next()
            for epoch in range(self.max_epochs):
                np.random.shuffle(data)
                batch_x = data[:, : self._dim_input]
                batch_y = data[:, self._dim_input:]
                #(batch_x, batch_y)= self._sess.run(train_loader)
                
                feed_dict = {self._meta_train_x: batch_x,
                             self._meta_train_y: batch_y,
                             }
                # print('pre \nweight[w1] ', self._weights['w1'].eval(session=self._sess))
                _, summary_str, meta_train_loss = \
                    self._sess.run([self._meta_train_op, self._summary_op, self._meta_train_loss,], feed_dict)

            self._writer.add_summary(summary_str)
            return meta_train_loss
                # print('post \nweight[w1] ', self._weights['w1'].eval(session=self._sess))
        

    def predict(self, state, action):
        """
        Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model

        :param states:    numpy array (lengthOfRollout , state.shape[1] )
        :param actions:
        :return:
        """
        """ YOUR CODE HERE """
    
        x_data = np.concatenate((state, action), axis = 1)
        #x_data = x_data[np.newaxis, :]
        f , cost = \
            self._sess.run([ self._meta_train_output, self._model_cost ],
                           {self._meta_train_x: x_data, self._meta_train_y: state ,})
        
        next_states = f+ state
    
        return next_states, cost

    
    def _traj_hc_cost( self, tf_x, tf_y, weights=1.0, scope=None):

        c_x = tf_x
        c_y = tf.add(tf_y, c_x[:, :self._dim_output])
    
        action = c_x[:, self._dim_output:]   # 20:
        n1 = tf.multiply(tf.constant(0.1), tf.reduce_sum(tf.pow(action, 2), axis=1))
        n2 = tf.div(tf.subtract(c_y[:, 17], c_x[:, 17]), tf.constant(-0.01))
        cost  =  tf.add(n1, n2)
        return cost

    def _traj_ant_cost(self, tf_x, tf_y, weights=1.0, scope=None):
    
        c_x = tf_x
        c_y = tf.add(tf_y, c_x[:, :self._dim_output])
    
        action = c_x[:, self._dim_output:]  # 20:
        n1 = tf.multiply(tf.constant(0.005), tf.reduce_sum(tf.pow(action, 2), axis=1))
        n2 = tf.div(tf.subtract(c_y[:, 27], c_x[:, 27]), tf.constant(-0.01))
        cost = tf.add(tf.add(n1, n2),tf.constant(0.05))
        return cost