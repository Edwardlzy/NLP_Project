from tensor2tensor import problems
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
import json
from tensor2tensor.utils import usr_dir


PROBLEM = 'translate_ende_wmt32k'
TMP_DIR = './raw_data/' # Where data files from internet stored
DATA_DIR = './processed_data/' # Where pre-prcessed data is stored
MODEL = 'transformer'
HPARAMS = 'transformer_gpt2'
TRAIN_DIR = './model_files_default/'

usr_dir.import_usr_dir("/h/edwardlzy/NLP_Project/tensor2tensor/tensor2tensor/data_generators/openwebtext/")

# Init problem T2T object the generated training data
t2t_problem = problems.problem(PROBLEM)
# t2t_problem.generate_data(DATA_DIR, TMP_DIR) 

# Print all models in T2T to console
registry.list_models()

# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)

# Make Chngaes to Hparams
# hparams.batch_size = 1024
# hparams.learning_rate_warmup_steps = 4000 #45000
# hparams.learning_rate = .4

# Can see all Hparams with code below
print(hparams)
print(json.loads(hparams.to_json()))
