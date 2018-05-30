import tensorflow as tf
import numpy as np
import os.path as osp, shutil, time, atexit, os, subprocess
import shutil
import glob
import csv
from datetime import datetime
import pandas as pd


def contruct_layer(inp, activation_fn, reuse, norm, is_train, scope):
    if norm == 'batch_norm':
        out = tf.contrib.layers.batch_norm(inp, activation_fn=activation_fn,
                                           reuse=reuse, is_training=is_train,
                                           scope=scope)
    elif norm == 'None':
        out = activation_fn(inp)
    else:
        ValueError('Can\'t recognize {}'.format(norm))
    return out


def get_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1/10.
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)


def configure_log_dir(logname, txt='', copy = True):
	"""
	Set output directory to d, or to /tmp/somerandomnumber if d is None
	"""

	now = datetime.now().strftime("%b-%d_%H:%M:%S")
	path = os.path.join('log-files', logname, now + txt)
	os.makedirs(path)  # create path
	if copy:
		filenames = glob.glob('*.py')  # put copy of all python files in log_dir
		for filename in filenames:  # for reference
			shutil.copy(filename, path)
	return path

class Logger(object):
	""" Simple training logger: saves to file and optionally prints to stdout """
	def __init__(self, logdir,csvname = 'log'):
		"""
		Args:
			logname: name for log (e.g. 'Hopper-v1')
			now: unique sub-directory name (e.g. date/time string)
		"""
		self.path = os.path.join(logdir, csvname+'.csv')
		self.write_header = True
		self.log_entry = {}
		self.f = open(self.path, 'w')
		self.writer = None  # DictWriter created with first call to write() method

	def write(self, display=True):
		""" Write 1 log entry to file, and optionally to stdout
		Log fields preceded by '_' will not be printed to stdout

		Args:
			display: boolean, print to stdout
		"""
		if display:
			self.disp(self.log_entry)
		if self.write_header:
			fieldnames = [x for x in self.log_entry.keys()]
			self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
			self.writer.writeheader()
			self.write_header = False
		self.writer.writerow(self.log_entry)
		self.log_enbtry = {}

	@staticmethod
	def disp(log):
		"""Print metrics to stdout"""
		log_keys = [k for k in log.keys()]
		log_keys.sort()
		'''
		print('***** Episode {}, Mean R = {:.1f} *****'.format(log['_Episode'],
															   log['_MeanReward']))
		for key in log_keys:
			if key[0] != '_':  # don't display log items with leading '_'
				print('{:s}: {:.3g}'.format(key, log[key]))
		'''
		print('log writed!')
		print('\n')

	def log(self, items):
		""" Update fields in log (does not write to file, used to collect updates.

		Args:
			items: dictionary of items to update
		"""
		self.log_entry.update(items)

	def close(self):
		""" Close log file - log cannot be written after this """
		self.f.close()

	def log_table2csv(self,data, header = True):
		df = pd.DataFrame(data)
		df.to_csv(self.path, index=False, header=header)


	def log_csv2table(self):
		data = pd.read_csv(self.path,header = 0,encoding='utf-8')
		return np.array(data)