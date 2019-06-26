import subprocess, os, sys
from itertools import product
import argparse, signal, datetime
from launcher_utils import *
import re

parser = argparse.ArgumentParser()
parser.add_argument('--batch_id', type=int, default=0)
parser.add_argument('--num_cpus', type=int, default=2)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--mem', type=int, default=16)
parser.add_argument('--partition', type=str, default='gpu')
parser.add_argument('--job_id', type=int, default=-1)
parser.add_argument('--binary', type=str, default='t2t-trainer')
parser.add_argument('--local_gpu_id', type=int, default=-1)
parser.add_argument('--username', type=str, default='jba')
parser.add_argument('--batch_config', type=str, default='')
parser.add_argument('--exclude', type=str, default='')
parser.add_argument('--noautosave', dest='noautosave', action='store_true')
parser.add_argument('--local', dest='local', action='store_true')
parser.add_argument('--interactive', dest='interactive', action='store_true')
parser.add_argument('--cslab', dest='cslab', action='store_true')
parser.add_argument('--nogobi', dest='nogobi', action='store_true')
parser.add_argument('--stats', dest='stats', action='store_true')
parser.add_argument('--submit', dest='submit', action='store_true')
parser.add_argument('--dry_run', dest='dry_run', action='store_true')
parser.add_argument('--cancel_all', dest='cancel_all', action='store_true')
parser.add_argument('--cancel_confirm', dest='cancel_confirm', action='store_true')
# Arguments for distributed training
parser.add_argument('--distributed', dest='distributed', action='store_true')
parser.add_argument('--master_address', type=str, default='gpu012:5555')
parser.add_argument('--worker_address', type=str, default='gpu042:5555,gpu045:5555', help='Comma-separated node name(s).')
parser.add_argument('--master_args_path', type=str, default='./example_master_arguments.txt')
parser.add_argument('--worker_args_path', type=str, default='./example_worker_arguments.txt')
parser.add_argument('--num_gpus_per_worker', type=int, default=4, help='Number of GPU per worker. (8 on q cluester, 4 on Vaughan cluster)')
parser.add_argument('--num_cpus_per_worker', type=int, default=8, help='Number of CPU per worker.')
parser.add_argument('--mem_per_worker', type=int, default=48)
parser.add_argument('--master_mem', type=int, default=32)
parser.add_argument('--master_num_cpus', type=int, default=8)
parser.add_argument('--is_q', dest='is_q', action='store_true', help='Whether on MaRS or Vaughan cluster')
parser.add_argument('--async', dest='async', action='store_true', help='Whether to launch synchronous or asynchronous distributed training.')

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

if FLAGS.distributed:
  # Here for tf_config
  # master_gpu = FLAGS.master_address.split(':')[0]
  # master_partition = get_partition(master_gpu, FLAGS.is_q)

  masters = FLAGS.master_address.split(',')
  num_masters = len(masters)
  workers = FLAGS.worker_address.split(',')
  num_workers = len(workers)

  # Generate TF_CONFIG
  MAKE_TF_CONFIGS = "--masters={} --ps={}".format(FLAGS.master_address, FLAGS.worker_address)
  
  # Setup master(s)
  MASTER_TF_CONFIG = []
  MASTER_SLURM_CMD = []
  for i in range(num_masters):
    cur_master = masters[i].split(':')[0]
    if not FLAGS.async:
      cur_master_tf_config = 'TF_CONFIG=\'{"cluster": {"master": {}, "ps": {}}, "environment": "cloud", "task": {"index": {}, "type": "master"}}\';'.format(FLAGS.master_address.split(','), FLAGS.worker_address.split(','), i)
    else:
      if i == 0:
        cur_master_tf_config = 'TF_CONFIG=\'{"task": {"index": 0, "type": "chief"}, "cluster": {"chief": {}, "ps": {}, "worker": {}}, "environment": "cloud"}\''.format([masters[0]], workers, masters[1:])
      else:
        cur_master_tf_config = 'TF_CONFIG=\'{"task": {"index": 0, "type": "worker"}, "cluster": {"chief": {}, "ps": {}, "worker": {}}, "environment": "cloud"}\''.format([masters[0]], workers, masters[1:])
    cur_master_slurm_cmd = "srun --mem {}G --gres=gpu:1 -c {} -w {} -p {} ".format(FLAGS.master_mem, FLAGS.master_num_cpus, cur_master, get_partition(cur_master, FLAGS.is_q))
    MASTER_TF_CONFIG.append(cur_master_tf_config)
    MASTER_SLURM_CMD.append(cur_master_slurm_cmd)

  # TF_CONFIG_CMD = "srun --gres=gpu:1 -c 4 --mem=8G -w {} -p {} t2t-make-tf-configs --masters={} --ps={}".format(master_gpu, master_partition, FLAGS.master_address, FLAGS.worker_address)
  # MAKE_TF_CONFIGS = "t2t-make-tf-configs --masters={} --ps={}".format(FLAGS.master_address, FLAGS.worker_address)
  # MASTER_TF_CONFIG = 'export TF_CONFIG=\'{"cluster": {"master": [{}], "ps": {}}, "environment": "cloud", "task": {"index": 0, "type": "master"}}\';'.format(FLAGS.master_address, FLAGS.worker_address.split(','))
  # MASTER_SLURM_CMD = "srun --mem {}G --gres=gpu:1 -c {} -w {} -p {} ".format(FLAGS.master_mem, FLAGS.master_num_cpus, master_gpu, master_partition)

  WORKER_TF_CONFIG = []
  WORKER_SLURM_CMD = []
  
  for i in range(num_workers):
    cur_worker_node = workers[i].split(':')[0]
    if not FLAGS.async:
      cur_worker_tf_config = 'TF_CONFIG=\'{"cluster": {"master": [{}], "ps": {}}, "environment": "cloud", "task": {"index": {}, "type": "ps"}}\''.format(FLAGS.master_address, FLAGS.worker_address.split(','), i)
    else:
      cur_worker_tf_config = 'TF_CONFIG=\'{"task": {"index": {}, "type": "ps"}, "cluster": {"chief": {}, "ps": {}, "worker": {}}, "environment": "cloud"}\''.format(i, [masters[0]], workers, masters[1:])
    cur_worker_slurm_cmd = "srun --mem {}G --gres=gpu:{} -c {} -w {} -p {} ".format(FLAGS.mem_per_worker, FLAGS.num_gpus_per_worker, FLAGS.num_cpus_per_worker, cur_worker_node, get_partition(cur_worker_node, FLAGS.is_q))
    WORKER_TF_CONFIG.append(cur_worker_tf_config)
    WORKER_SLURM_CMD.append(cur_worker_slurm_cmd)

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
  if FLAGS.distributed:
    jobs = create_distributed_jobs(FLAGS.job_id, is_master=True)
    jobs += create_distributed_jobs(FLAGS.job_id)
  else:
    jobs = create_jobs(FLAGS.job_id)

  if FLAGS.stats:
    for job in jobs:
      get_log_file(job.get_save_dir())

  if FLAGS.submit or FLAGS.dry_run:
    for job in jobs:
      job.start()

    if FLAGS.submit and FLAGS.interactive:
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


def script_command(binary, exp_name, hyperparams, gpu_id, slurm_cmd=SLURM_CMD, is_bash=False):
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
    LAUNCH_CMD = slurm_cmd + '-J {}'.format(JOB_ID)
  else:
    # on local workstation
    JOB_ID = "CUDA_VISIBLE_DEVICES="+ str(LOCAL_GPU_ID[gpu_id])
    LAUNCH_CMD = JOB_ID
  if not is_bash:
    script_command_list = [PYTHON_CMD, 
                           binary,  
                           hyperparams, 
                          ] 
  else:
    script_command_list = [binary, hyperparams]
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


def create_distributed_jobs(job_id, is_master=False):
  """ Creates a list of jobs of master and workers. """
  jobs = []
  GPU_ID_COUNT = 0

  if is_master:
    # Setup TF_CONFIG first.
    cmd, job_id_str, save_dir = script_command('t2t-make-tf-configs', EXP_NAME, MAKE_TF_CONFIGS, GPU_ID_COUNT, MASTER_SLURM_CMD[0])
    print(cmd)
    jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))

    with open(FLAGS.master_args_path) as f:
      partial_master_args = f.read()

    for i in range(num_masters):
      # Build the hyperparameters for the current master node.
      if FLAGS.async:
        worker_job = '/job:chief' if i == 0 else '/job:worker'
      else:
        worker_job = '/job:master'
      master_args = "--master=grpc://{} --ps_replicas={} --worker_replicas={} --worker_gpu=1 --worker_id={} --ps_gpu={} --worker_job={} ".format(masters[i], num_workers, num_masters, i, FLAGS.num_gpus_per_worker, worker_job)
      if not FLAGS.async: master_args += '--sync '
      master_args += partial_master_args

      # Export TF_CONFIG.
      cmd, job_id_str, save_dir = script_command('export', EXP_NAME, MASTER_TF_CONFIG[i], GPU_ID_COUNT, MASTER_SLURM_CMD[i], True)
      print(cmd)
      jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))

      # Launch the master.
      cmd, job_id_str, save_dir = script_command(FLAGS.binary, EXP_NAME, master_args, GPU_ID_COUNT, MASTER_SLURM_CMD[i])
      print(cmd)
      jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))

  else:
    with open(FLAGS.worker_args_path) as f:
      worker_args = f.read()
    for i in range(num_workers):
      # Export TF_CONFIG.
      cmd, job_id_str, save_dir = script_command('export', EXP_NAME, WORKER_TF_CONFIG[i], GPU_ID_COUNT, WORKER_SLURM_CMD[i], True)
      print(cmd)
      jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))

      # Launch the worker.
      cmd, job_id_str, save_dir = script_command(FLAGS.binary, EXP_NAME, worker_args, GPU_ID_COUNT, WORKER_SLURM_CMD[i])
      print(cmd)
      jobs.append(job_instance(cmd, job_id_str, save_dir, FLAGS))

  if job_id == -1:
    return jobs
  else:
    return [jobs[job_id],]


if __name__ == '__main__':
  main(FLAGS)
