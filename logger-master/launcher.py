import subprocess, os, sys
from itertools import product
import argparse, signal, datetime
from launcher_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_id', type=int, default=0)
parser.add_argument('--num_cpus', type=int, default=2)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--mem', type=int, default=16)
parser.add_argument('--partition', type=str, default='gpu')
parser.add_argument('--job_id', type=int, default=-1)
parser.add_argument('--binary', type=str, default='')
parser.add_argument('--local_gpu_id', type=int, default=-1)
parser.add_argument('--username', type=str, default='jba')
parser.add_argument('--batch_config', type=str, default='')
parser.add_argument('--exclude', type=str, default='')
parser.add_argument('--noautosave', dest='noautosave', action='store_true')
parser.add_argument('--local', dest='local', action='store_true')
parser.add_argument('--interactive', dest='interative', action='store_true')
parser.add_argument('--cslab', dest='cslab', action='store_true')
parser.add_argument('--nogobi', dest='nogobi', action='store_true')
parser.add_argument('--stats', dest='stats', action='store_true')
parser.add_argument('--submit', dest='submit', action='store_true')
parser.add_argument('--dry_run', dest='dry_run', action='store_true')
parser.add_argument('--cancel_all', dest='cancel_all', action='store_true')
parser.add_argument('--cancel_confirm', dest='cancel_confirm', action='store_true')

global FLAGS
FLAGS, extraFLAGS = parser.parse_known_args()

## one-off experiments using '--'
if FLAGS.binary == '' and extraFLAGS[0] == '--':
  FLAGS.binary = extraFLAGS[1]
  extraFLAGS = extraFLAGS[2:]
  EXP_NAME = os.path.splitext(FLAGS.binary)[0]

assert FLAGS.batch_config != '' or FLAGS.binary != '', 'config file or binary cannot be empty'
if FLAGS.batch_config != '':
  execfile(FLAGS.batch_config)

# Here for tf_config

# Add case for worker & master setup command.

if FLAGS.cslab:
  SLURM_CMD = "srun --mem {}gb --gres=gpu:{} -c {} -l -x {} -p {}c ".format(FLAGS.mem, FLAGS.num_gpus, FLAGS.num_cpus, "guppy10,guppy[13-22]", FLAGS.partition)
else:
  if FLAGS.exclude != '':
    SLURM_CMD = "srun --unbuffered --mem {}gb --gres=gpu:{} -c {} -l -x {} -p {} ".format(FLAGS.mem, FLAGS.num_gpus, FLAGS.num_cpus, FLAGS.exclude, FLAGS.partition)
  else:
    SLURM_CMD = "srun --unbuffered --mem {}gb --gres=gpu:{} -c {} -l -p {} ".format(FLAGS.mem, FLAGS.num_gpus, FLAGS.num_cpus, FLAGS.partition)

HOME_DIR = os.getenv("HOME")
if FLAGS.local:
  if FLAGS.local_gpu_id != -1:
    LOCAL_GPU_ID = [FLAGS.local_gpu_id, ]
  else:
    LOCAL_GPU_ID = get_remaining_local_gpu_id()

PYTHON_CMD = sys.executable
BASE_SAVE_DIR = os.path.join(HOME_DIR, "gobi_local", EXP_NAME)
job_instance = local_job if FLAGS.local else slurm_job

def main(_):
  jobs = create_jobs(FLAGS.job_id)

  if FLAGS.stats:
    for job in jobs:
      get_log_file(job.get_save_dir())

  if FLAGS.submit or FLAGS.dry_run:
    for job in jobs:
      job.start()

    if FLAGS.submit and FLAGS.interative:
      try:
        job.interact()
      except KeyboardInterrupt:
        job.cancel()

  if FLAGS.cancel_all:
    for job in jobs:
      job.cancel()

def hyperParamIterator():
  ### loop over permutations of all the hyper-params
  search_params = {k:SEARCH_PARAMS[k] for k in SEARCH_PARAMS if k != 'model'}
  search_params[' '] = SEARCH_PARAMS['model']
  permutation_list = list(product(*search_params.values()))
  for p_params in permutation_list:
    hyperparams = ' '.join([' '.join([i, str(j)]) for i,j in zip(search_params.keys(), p_params)])
    yield hyperparams.strip(' ')

def script_command(binary, exp_name, hyperparams, gpu_id):
  hyperparams_name = hyperparams.replace('-','').replace(' ','_').replace('.','')
  save_name = '_'.join([exp_name, hyperparams_name])
  if len(extraFLAGS) > 0:
    extra_flags_name = " ".join(extraFLAGS).replace(' ','_').replace('-','').replace('.','')
    save_name += '_' + extra_flags_name
  if FLAGS.nogobi:
    save_dir = save_name
  else:
    save_dir = os.path.join(BASE_SAVE_DIR, save_name)
  if not FLAGS.local:
    # on slurm
    if FLAGS.batch_config != '':
      # batch-mode on slurm
      JOB_ID = str(BATCH_ID)+'/'+str(gpu_id)
    else:
      # one-off on slurm
      JOB_ID = str(datetime.datetime.now().strftime("%d%H%M%S"))
    LAUNCH_CMD = SLURM_CMD + '-J {}'.format(JOB_ID)
  else:
    # on local workstation
    JOB_ID = "CUDA_VISIBLE_DEVICES="+ str(LOCAL_GPU_ID[gpu_id])
    LAUNCH_CMD = JOB_ID
  script_command_list = [PYTHON_CMD, 
                         binary,  
                         hyperparams, 
                        ] 
  if not FLAGS.noautosave:
    script_command_list += ["--save_name", save_name, 
                            "--save_dir", save_dir,]
  script_command_list += extraFLAGS
  if FLAGS.batch_config != '':
    # batch-mode: add in hyperparams we are not searching over
    script_command_list += EXTRA_PARAM

  script_command = LAUNCH_CMD + " " + " ".join(script_command_list)

  return script_command, JOB_ID, save_dir

def create_jobs(job_id):
  jobs = []
  GPU_ID_COUNT=0
  if FLAGS.batch_config != '':
    ### batch-mode: loop over permutations of all the hyper-params
    for hyperparams in hyperParamIterator():
      cmd, job_id_str, save_dir = script_command(BINARY, EXP_NAME, hyperparams, GPU_ID_COUNT)
      print(cmd)
      jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))
      GPU_ID_COUNT += 1
  elif FLAGS.binary != '':
    ### one-off mode: 
    cmd, job_id_str, save_dir = script_command(FLAGS.binary, EXP_NAME, '', GPU_ID_COUNT)
    print(cmd)
    jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))

  if job_id == -1:
    return jobs
  else:
    return [jobs[job_id],]


if __name__ == '__main__':
  main(FLAGS)
