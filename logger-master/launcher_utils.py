import subprocess, os
import argparse
import numpy as np
try:
    import cPickle as pkl
    import commands
except ImportError:
    # Python 3 pickle import
    import _pickle as cPickle
    import subprocess as commands

global FLAGS

def get_remaining_local_gpu_id():
  sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  gpu_list_str = sp.communicate()
  num_gpus = len(gpu_list_str[0].split('\n')[:-1])
  status_code, processes_list_str = commands.getstatusoutput('''nvidia-smi | awk '$4=="C" {print $2}' ''')
  assert status_code == 0
  processes_list = processes_list_str.split('\n')
  processes_list = [int(i) for i in processes_list if len(i) > 0]
  remaining_gpus = [i for i in range(num_gpus) if i not in processes_list]
  return remaining_gpus


def get_log_file(path):
  log_file_name = os.path.join(path, 'log.pkl')
  if os.path.exists(log_file_name):
    with open(log_file_name, 'rb') as fp:
      log = pkl.load(fp)
    candidates = [log[k] for k in log if 'inception' in k]
    if len(candidates) == 0:
      print("cound not find the field")
    else:
      inception_scores = sorted(candidates[0].items())
      inception_scores = zip(*inception_scores)[1]
      tenk_inception_score = inception_scores[-1] if len(inception_scores) < 10 else inception_scores[9] 
      max_inception_score, max_inception_score_idx = np.max(inception_scores), np.argmax(inception_scores)
      inception_score_std = np.std(inception_scores[-5:])
      if True:
        print(path)
        print("inception@")
        print(inception_scores)
        print("1k: {:.2f}".format(inception_scores[0]))
        if len(inception_scores) > 5:
          print("5k: {:.2f}".format(inception_scores[5]))
        if len(inception_scores) > 50:
          print("50k: {:.2f}".format(inception_scores[50]))
        print("lat: {:.2f}".format(inception_scores[-1]))
        print("max: {:.2f} iter:{}".format(max_inception_score, max_inception_score_idx))
        print("std: {:.2f} iter:{}".format(inception_score_std, len(inception_scores)))
        #  return None
  else:
    return None

class Job(object):
  def __init__(self, cmd, job_id, save_dir, args):
     self.cmd = cmd
     self.job_id = job_id
     self.save_dir = save_dir
     self.args = args
     self.process = None

  def get_cmd(self):
     return self.cmd

  def get_save_dir(self):
     return self.save_dir

  def start(self,):
     env_vars = os.environ.copy()
     if self.args.dry_run:
       return None
     if self.args.submit:
       self.process = subprocess.Popen(self.cmd, env=env_vars, shell=True)
       return self.process

  def interact(self,):
     return self.process.communicate()

  def cancel(self, job_id): 
     raise NotImplementedError



class slurm_job(Job):
  def __init__(self, cmd, job_id, save_dir, args):
     super(slurm_job, self).__init__(cmd, job_id, save_dir, args)

  def cancel(self):
     squeue_output = os.popen("squeue -u {}".format(self.args.username) + " | awk '{print $1 \" \" $3}'").readlines()
     squeue_jobs = [i.strip("\n").split(" ") for i in squeue_output if 'JOBID' not in i]
     candidates = [s_id for s_id, job_id in squeue_jobs if self.job_id == job_id]
     if len(candidates) > 0:
       slurm_id = candidates[0]
       slurm_cancel_cmd = "scancel {}".format(slurm_id)
       print(slurm_cancel_cmd)
       if self.args.cancel_confirm or self.args.interative:
         os.popen(slurm_cancel_cmd)
       else:
         print("are you sure to cancel those jobs? use --cancel_confirm")
     else:
         print("!!!!!!!WARNING: {} not found".format(self.job_id))
       
      
class local_job(Job):
  def __init__(self, cmd, job_id, save_dir, args):
     super(local_job, self).__init__(cmd, job_id, save_dir, args)

  def cancel(self):
     ps_output = os.popen("ps u".format(self.args.username) + " | awk '{print $2 \" \" $13}'").readlines()
     local_jobs = [i.strip("\n").split(" ") for i in ps_output if 'PID' not in i]
     pid = [pid for pid, job_cmd in local_jobs if self.job_id in job_cmd][0]
     local_cancel_cmd = "kill -9 {}".format(pid)
     print(local_cancel_cmd)
     if self.args.cancel_confirm or self.args.interative:
       os.system(local_cancel_cmd)
     else:
       print("are you sure to cancel those jobs? use --cancel_confirm")
    
