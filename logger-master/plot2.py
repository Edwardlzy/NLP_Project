import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io.file_io import recursive_create_dir
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
from datetime import datetime
import cPickle as pickle
import os

FLAGS = tf.flags.FLAGS

def check_dir(path):
  if not tf.gfile.Exists(path):
    recursive_create_dir(path)

class plotter():
  def __init__(self, save_dir, name):
    self._save_dir = save_dir
    self._name = name
    self._since_beginning = collections.defaultdict(lambda: {})
    self._since_last_flush = collections.defaultdict(lambda: {})
    self._since_last_flush_ordered_key = []
    self._iter = [0]

  def logger_hook(self, print_dict, print_freq=100):
    plot = self.plot
    tick = self.tick
    flush = self.flush
    name = self._name
    class _LoggerHook(tf.train.SessionRunHook):
        """Logs things."""
        def begin(self):
          self._step = -1
          self._print_freq = print_freq
          self._print_dict = print_dict
          self._print_names, self._print_tensors = zip(*self._print_dict.iteritems())
          self._start_time = time.time()
        def before_run(self, run_context):
          self._step += 1
          return tf.train.SessionRunArgs(self._print_tensors)
        def after_run(self, run_context, run_values):
          tick()
          results = run_values.results
          global_step = 0
          ## store each monitored tensor
          for k, v in zip(self._print_names, results):
            if k != 'global_step':
              plot(k, v)
            else:
              global_step = v    
          ## print monitored tensor
          if self._step % self._print_freq == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            sec_per_batch = float(self._print_freq/duration)
            print ('%s: %s: step %d, %.3f batch/sec' % (datetime.now(), name, 
                                                        global_step, sec_per_batch))
            flush()
  
    return _LoggerHook()
  
  
  def tick(self,):
  	self._iter[0] += 1
  
  def plot(self, name, value):
    if not (name in self._since_last_flush_ordered_key):
      self._since_last_flush_ordered_key.append(name)
    self._since_last_flush[name][self._iter[0]] = value
  
 
  def save_log(self,):
    LOG_FILE = '{}/log.pkl'.format(self._save_dir)
    with open(LOG_FILE, 'wb') as f:
      pickle.dump(dict(self._since_beginning), f, pickle.HIGHEST_PROTOCOL)
  
  def save_figure(self, name):
    save_name = name.replace(' ', '_')+'.jpg'
    plt.savefig('{}/{}'.format(self._save_dir, save_name))
  
  def flush(self, ):
    check_dir(self._save_dir)
  
    prints = []
    for name in self._since_last_flush_ordered_key:
      vals = self._since_last_flush[name]
      prints.append("{}\t{}".format(name, np.mean(vals.values())))
      self._since_beginning[name].update(vals)
      
      x_vals = np.sort(self._since_beginning[name].keys())
      y_vals = [self._since_beginning[name][x] for x in x_vals]
      
      plt.clf()
      plt.plot(x_vals, y_vals)
      plt.xlabel('iteration')
      plt.ylabel(name)
      self.save_figure(name)
    
    print "iter {}\t{}".format(self._iter[0], "\t".join(prints))
    self._since_last_flush.clear()
    self._since_last_flush_ordered_key = []
  
    self.save_log()
