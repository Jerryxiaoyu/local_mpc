import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf

import utils
from loginfo import log
from utils import *

# global variables for MAML
LOG_FREQ = 10
SUMMARY_FREQ = 1
SAVE_FREQ = 10
EVAL_FREQ = 1


class MAML(object):
    def __init__(self, env,dataset, model_type, loss_type, dim_input, dim_output,
                 alpha, beta, K, num_paths,batch_size, is_train,num_updates, norm,sample_task_fun,task_Note = 'a', task_range=None, logdir = 'log/'):
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
        
        self._dataset = dataset
        self._alpha = alpha
        self._K = K
        self._norm = norm
        self._dim_input = dim_input
        self._dim_output = dim_output
        self._batch_size = batch_size
        
        self._num_updates = num_updates
        self._meta_optimizer = tf.train.AdamOptimizer(beta)
        self._avoid_second_derivative = False
        self._task_Note = task_Note
        self._task_name = 'MAML.{}_{}_{}-shot_{}-updates_{}-batch_norm-{}-EXP_{}'.format(env,dataset.name, self._K,
                                                                               self._num_updates, self._batch_size,
                                                                               self._norm,self._task_Note)
        log.infov('Task name: {}'.format(self._task_name))
        self._logdir = logdir
        
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
        self._summary_dir = os.path.join(self._logdir,'log', self._task_name)
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
        
        print('MAML Init')
        print('weight[w1] ', self._weights['w1'].eval(session=self._sess))

        self.num_paths = num_paths
        #self.split_val = num_paths_random *env_horizon *0.8
        #self.split_val= int(self.num_paths*self._K*0.5)
      
        self._sample_task_fun = sample_task_fun
        self._task_range = task_range

    def _build_placeholder(self):
        self._meta_train_x = tf.placeholder(tf.float32)
        self._meta_train_y = tf.placeholder(tf.float32)
        self._meta_val_x = tf.placeholder(tf.float32)
        self._meta_val_y = tf.placeholder(tf.float32)

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

        # Calculate loss on 1 task
        def metastep_graph(inp):
            meta_train_x, meta_train_y, meta_val_x, meta_val_y = inp
            meta_train_loss_list = []
            meta_val_loss_list = []

            weights = self._weights
            meta_train_output = self._contruct_forward(meta_train_x, weights,
                                                       reuse=False, norm=norm,
                                                       is_train=self._is_train)
            # Meta train loss: Calculate gradient
            meta_train_loss = self._loss_fn(meta_train_y, meta_train_output)
            meta_train_loss = tf.reduce_mean(meta_train_loss)
            meta_train_loss_list.append(meta_train_loss)
            grads = dict(zip(weights.keys(),
                         tf.gradients(meta_train_loss, list(weights.values()))))
            new_weights = dict(zip(weights.keys(),
                               [weights[key]-self._alpha*grads[key]
                                for key in weights.keys()]))
            if self._avoid_second_derivative:
                new_weights = tf.stop_gradients(new_weights)
            meta_val_output = self._contruct_forward(meta_val_x, new_weights,
                                                     reuse=True, norm=norm,
                                                     is_train=self._is_train)
            # Meta val loss: Calculate loss (meta step)
            meta_val_loss = self._loss_fn(meta_val_y, meta_val_output)
            meta_val_loss = tf.reduce_mean(meta_val_loss)
            meta_val_loss_list.append(meta_val_loss)
            # If perform multiple updates
            for _ in range(self._num_updates-1):
                meta_train_output = self._contruct_forward(meta_train_x, new_weights,
                                                           reuse=True, norm=norm,
                                                           is_train=self._is_train)
                meta_train_loss = self._loss_fn(meta_train_y, meta_train_output)
                meta_train_loss = tf.reduce_mean(meta_train_loss)
                meta_train_loss_list.append(meta_train_loss)
                grads = dict(zip(new_weights.keys(),
                                 tf.gradients(meta_train_loss, list(new_weights.values()))))
                new_weights = dict(zip(new_weights.keys(),
                                       [new_weights[key]-self._alpha*grads[key]
                                        for key in new_weights.keys()]))
                if self._avoid_second_derivative:
                    new_weights = tf.stop_gradients(new_weights)
                meta_val_output = self._contruct_forward(meta_val_x, new_weights,
                                                         reuse=True, norm=norm,
                                                         is_train=self._is_train)
                meta_val_loss = self._loss_fn(meta_val_y, meta_val_output)
                meta_val_loss = tf.reduce_mean(meta_val_loss)
                meta_val_loss_list.append(meta_val_loss)

            return [meta_train_loss_list, meta_val_loss_list, meta_train_output, meta_val_output]

        output_dtype = [[tf.float32]*self._num_updates, [tf.float32]*self._num_updates,
                        tf.float32, tf.float32]
        # tf.map_fn: map on the list of tensors unpacked from `elems`
        #               on dimension 0 (Task)
        # reture a packed value
        result = tf.map_fn(metastep_graph,
                           elems=(self._meta_train_x, self._meta_train_y,
                                  self._meta_val_x, self._meta_val_y),
                           dtype=output_dtype, parallel_iterations=self._batch_size)
        meta_train_losses, meta_val_losses, meta_train_output, meta_val_output = result
        self._meta_val_output = meta_val_output
        self._meta_train_output = meta_train_output
        self.total_loss1 = total_loss1 =   tf.reduce_sum(meta_train_losses) / tf.to_float(self._batch_size)
        self.total_losses2 =  [tf.reduce_sum(meta_val_losses[j]) / tf.to_float(self._batch_size) for j in
                                              range(self._num_updates)]
        self._pretrain_op = self._meta_optimizer.minimize(total_loss1)
        
        # Only look at the last final output
        meta_train_loss = tf.reduce_mean(meta_train_losses[-1])
        meta_val_loss  = tf.reduce_mean(meta_val_losses[-1])


        # Loss
        self._meta_train_loss = meta_train_loss
        self._meta_val_loss = meta_val_loss
        # Meta train step
        self._meta_train_op = self._meta_optimizer.minimize(meta_val_loss )


        
        # Summary
        self._pre_train_loss_sum = tf.summary.scalar('loss/pre_train_loss', total_loss1)
        self._meta_train_loss_sum = tf.summary.scalar('loss/meta_train_loss', meta_train_loss)
       # self._meta_val_loss_sum = tf.summary.scalar('loss/meta_val_loss', meta_val_loss)
        self._summary_op = tf.summary.merge_all()

    def learn(self, batch_size,  dataset, max_epochs, is_PreTrain=False, **kwargs):
        # collect data
        dataset.get_dataset(resample=True, task=None, controller='Rand', task_range=self._task_range, task_fun= self._sample_task_fun)
        data_x, data_y,LengthOfData = dataset.get_batch( batch_size)
        
        # load data
        data_x_placeholder = tf.placeholder(tf.float32, data_x.shape)
        data_y_placeholder = tf.placeholder(tf.float32, data_y.shape)
        dataset_tf = tf.data.Dataset.from_tensor_slices((data_x_placeholder, data_y_placeholder)).shuffle(
            buffer_size=10000).batch(batch_size).repeat()
        # create data iterator
        iter = dataset_tf.make_initializable_iterator()  # dataset.make_one_shot_iterator()
        train_loader = iter.get_next()
        # init data
        self._sess.run(iter.initializer, feed_dict={data_x_placeholder: data_x, data_y_placeholder:data_y})

        logger = Logger(self._logdir, csvname='log_loss')
        
        if is_PreTrain is True:
            input_tensors = [self._pretrain_op, self._summary_op, self._meta_val_loss, self._meta_train_loss, self.total_loss1, self.total_losses2]
        else:
            input_tensors = [self._meta_train_op, self._summary_op, self._meta_val_loss, self._meta_train_loss, self.total_loss1, self.total_losses2]
        for epoch in range(max_epochs):
            for i in range( int(LengthOfData / batch_size)):
                
                (batch_input, batch_target) = self._sess.run(train_loader)
    
                feed_dict = {self._meta_train_x: batch_input[:, :self._K, :],
                             self._meta_train_y: batch_target[:, :self._K, :],
                             self._meta_val_x: batch_input[:, self._K:, :],
                             self._meta_val_y: batch_target[:, self._K:, :]}
    
                _, summary_str, meta_val_loss, meta_train_loss, meta_loss1, meta_losses2 = \
                    self._sess.run(input_tensors, feed_dict)

                if i % LOG_FREQ == 0:
                    log.info("Epoch: {}/{} Step: {}/{}, Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
                        epoch, max_epochs,i, int(LengthOfData / batch_size), meta_train_loss, meta_val_loss))
                    
            # Log/TF_board/Save/Evaluate
            if epoch % SUMMARY_FREQ == 0:
                self._writer.add_summary(summary_str, epoch)
            # if epoch % LOG_FREQ == 0:
            #     log.info("Step: {}/{}, Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
            #         epoch, int(max_epochs), meta_train_loss, meta_val_loss))
            
            
            if (epoch+1) % SAVE_FREQ == 0:
                log.infov("Save checkpoint-{}".format(epoch))
                self._saver.save(self._sess, os.path.join(self._checkpoint_dir, 'checkpoint'),
                                 global_step=epoch)
            if (epoch+1) % EVAL_FREQ == 0:
                train_loss_mean, val_loss_mean = self.evaluate(dataset, 2, False, task_range=self._task_range)
                logger.log({'epoch': epoch,
                            'meta_val_loss': meta_val_loss,
                            'meta_train_loss':meta_train_loss,
                            'meta_loss1':meta_loss1,
                            'meta_loss2':meta_losses2,
                            'val_tain_loss_mean':train_loss_mean,
                            'val_val_loss_mean':val_loss_mean,
                            
                            })
                logger.write(display=False)
        #close logger
        logger.close()

    def evaluate(self, dataset, test_steps, draw, load_model=False,task_range=(0,7),task_type='rand',**kwargs):
        if load_model:
            assert kwargs['restore_checkpoint'] is not None or \
                kwargs['restore_dir'] is not None
            if kwargs['restore_checkpoint'] is None:
                restore_checkpoint = tf.train.latest_checkpoint(kwargs['restore_dir'])
            else:
                restore_checkpoint = kwargs['restore_checkpoint']
            self._saver.restore(self._sess, restore_checkpoint)
            log.infov('Load model: {}'.format(restore_checkpoint))

        accumulated_val_loss = []
        accumulated_train_loss = []
        for step in tqdm(range(test_steps)):
            if task_type =='rand':
                task = self._sample_task_fun(task_range[0], task_range[1] ,(1,))
            elif task_type =='lin':
                task = np.array([0 + step * 0.05, ])
            else:
                print('please check task type!')
         
            output, val_loss, train_loss ,x, y = self._single_test_step( dataset, num_tasks=1, task=task)
            
            if  load_model and draw:
                # visualize one by one
                draw_dir = os.path.join(self._logdir, 'vis', self._task_name,
                                        'exp_' + str(step) + '_task_num' + str(task[0]) + '_loss' + str(val_loss))
                if not os.path.exists(draw_dir):
                    os.makedirs(draw_dir)
                dataset.visualize(x[:, self._K:, :], y[:, self._K:, :], output,
                                  draw_dir)

            accumulated_val_loss.append(val_loss)
            accumulated_train_loss.append(train_loss)
        val_loss_mean = sum(accumulated_val_loss)/test_steps
        train_loss_mean = sum(accumulated_train_loss)/test_steps
        log.infov("[Evaluate] Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
            train_loss_mean, val_loss_mean))
        return train_loss_mean, val_loss_mean
        

    def _single_train_step(self,train_loader,input_tensors, batch_size ):
        
        
        #batch_input, batch_target  = dataset.get_batch(batch_size,  resample=True, num_paths_random =self.num_paths, task_range = self._task_range, task_fun =self._sample_task_fun)
        import math
        for i in range(math.ceil(len(data_x) / batch_size)):
            (batch_input, batch_target) = self._sess.run(train_loader)
        
            feed_dict = {self._meta_train_x: batch_input[:, :self.split_val, :],
                         self._meta_train_y: batch_target[:, :self.split_val, :],
                         self._meta_val_x: batch_input[:, self.split_val:, :],
                         self._meta_val_y: batch_target[:, self.split_val:, :]}
            
            _, summary_str, meta_val_loss, meta_train_loss , meta_loss1, meta_losses2= \
            self._sess.run(input_tensors, feed_dict)
        return meta_val_loss, meta_train_loss, summary_str


    def _single_test_step(self,dataset, num_tasks, task=None):
    
        batch_input, batch_target = dataset.get_test_batch(num_tasks=num_tasks, resample=True, task=task,

                                                           task_range=self._task_range,
                                                           task_fun=self._sample_task_fun)

        #(batch_input, batch_target) = self._sess.run(test_loader)
        feed_dict = {self._meta_train_x: batch_input[:, :self._K, :],
                     self._meta_train_y: batch_target[:, :self._K, :],
                     self._meta_val_x: batch_input[:, self._K:, :],
                     self._meta_val_y: batch_target[:, self._K:, :]}
        meta_val_output, meta_val_loss, meta_train_loss = \
            self._sess.run([self._meta_val_output, self._meta_val_loss,
                            self._meta_train_loss],
                           feed_dict)

        return meta_val_output, meta_val_loss, meta_train_loss, batch_input, batch_target
 
    def evaluate2(self, dataset, test_steps, draw, load_model=False,task_range=(0,7),**kwargs):
        '''
        evaluate meta-model over uniform distribution tasks to get loss
        
        :param dataset:
        :param test_steps:
        :param draw:
        :param load_model:
        :param task_range:
        :param kwargs:
        :return:
        '''
        if load_model:
            assert kwargs['restore_checkpoint'] is not None or \
                kwargs['restore_dir'] is not None
            if kwargs['restore_checkpoint'] is None:
                restore_checkpoint = tf.train.latest_checkpoint(kwargs['restore_dir'])
            else:
                restore_checkpoint = kwargs['restore_checkpoint']
            self._saver.restore(self._sess, restore_checkpoint)
            log.infov('Load model: {}'.format(restore_checkpoint))
        for tm in range(5):
            accumulated_val_loss = []
            accumulated_train_loss = []
            tasks =[]
            for step in tqdm(range(test_steps)):
                #task = self._sample_task_fun(task_range[0], task_range[1] ,(1,))
                task = np.array([0+step *0.05,])
                
                output, val_loss, train_loss, x ,y= self._single_test_step(dataset, 1, task=task)
                if  load_model and draw:
                    # visualize one by one
                    draw_dir = os.path.join(self._logdir,'vis', self._task_name, 'exp_'+str(step)+'_task_num'+str(task[0])+'_loss'+str(val_loss))
                    if not os.path.exists(draw_dir):
                        os.makedirs(draw_dir)
                    dataset.visualize(x[:, self._K:, :], y[:, self._K:, :], output,
                                      draw_dir)
                
                accumulated_val_loss.append(val_loss)
                accumulated_train_loss.append(train_loss)
                tasks.append(task)
        
            
            val_loss_mean = sum(accumulated_val_loss)/test_steps
            train_loss_mean = sum(accumulated_train_loss)/test_steps
            log.infov("[Evaluate] Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
                train_loss_mean, val_loss_mean))
            data_tmp =np.array(accumulated_val_loss).reshape((-1, 1))
            if tm ==0:
                data2 = data_tmp
            else:
                data2 = np.concatenate((data2, data_tmp), axis=1)
 
        log_dir = os.path.join('vis', self._task_name)
        logger = Logger(log_dir, csvname='log'  )
        data1 = np.array(tasks)
        
        data = np.concatenate((data1, data2), axis=1)
        logger.log_table2csv(data)
