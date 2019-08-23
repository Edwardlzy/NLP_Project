"""Data generators for OpenWebText data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from six.moves import range  # pylint: disable=redefined-builtin
import glob
import random
import traceback

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
from . import lookahead_tensorflow as lookahead_tf
from . import largebatch_optim
from tensor2tensor.models import transformer

import tensorflow as tf

@registry.register_optimizer
def lookahead(learning_rate, hparams):
  """By default, use LA_Adam with la_steps=10 and la_alpha=0.5."""
  optim = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=hparams.optimizer_adam_beta1,
            beta2=hparams.optimizer_adam_beta2,
            epsilon=hparams.optimizer_adam_epsilon)
  return lookahead_tf.LookaheadOptimizer(optim, 10)


@registry.register_optimizer
def largebatch(learning_rate, hparams):
  """ By default, use Adam with update_step=2. (Doubles the batch size) """
  optim = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=hparams.optimizer_adam_beta1,
            beta2=hparams.optimizer_adam_beta2,
            epsilon=hparams.optimizer_adam_epsilon)
  return largebatch_optim.LargebatchOptimizer(optim, 2)


@registry.register_hparams
def transformer_gpt2():
  """HParams for training gpt2 on OpenWebText."""
  hparams = transformer.transformer_lm_tpu_0()
  hparams.num_heads = 12  # Heads are expensive on TPUs.
  hparams.batch_size = 4096 #1024
  hparams.filter_size = 3072
  hparams.learning_rate_constant = 2.5
  hparams.hidden_size = 768
  hparams.learning_rate_warmup_steps = 2000
  hparams.learning_rate_minimum = 0.0
  hparams.learning_rate_cosine_cycle_steps = 2000000
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.max_length = 1024
  hparams.optimizer = "multistep_adam"
  hparams.optimizer_multistep_accumulate_steps = 32 #128
  hparams.num_hidden_layers = 12
  return hparams


split_files = None

def train_dev_split(tmp_dir, split, ratio=0.9, percentage=0.5):
  """Split the data into training and validation set."""
  global split_files
  if not split_files:
    if os.path.isfile(os.path.join(tmp_dir, 'training_set.txt')) and os.path.isfile(os.path.join(tmp_dir, 'val_set.txt')):
      tf.logging.info("Loading pre-generated splits...")
      f = open(os.path.join(tmp_dir, 'training_set.txt'), 'r')
      _train_data_filenames = f.read().split('\n')
      tf.logging.info("Using %d out of %d files.", round(len(_train_data_filenames) * percentage), len(_train_data_filenames))
      _train_data_filenames = _train_data_filenames[:round(len(_train_data_filenames) * percentage)]
      f = open(os.path.join(tmp_dir, 'val_set.txt'), 'r')
      _dev_data_filenames = f.read().split('\n')
    else:
      tf.logging.info("Generating train_val split...")
      dataset_filenames = glob.glob(os.path.join(tmp_dir, '*', '*.txt'))
      random.shuffle(dataset_filenames)
      training_num = round(len(dataset_filenames) * 0.9)
      _train_data_filenames = dataset_filenames[:training_num]
      _dev_data_filenames = dataset_filenames[training_num:]
    split_files = {
        problem.DatasetSplit.TRAIN: _train_data_filenames,
        problem.DatasetSplit.EVAL: _dev_data_filenames,
    }
  return split_files[split]


# def _train_data_filenames(tmp_dir):
#   # return [
#   #     os.path.join(tmp_dir,
#   #                  "openwebtext-language-modeling",
#   #                  "training-monolingual.bpe.shuffled",
#   #                  "text.en-%05d-of-00100" % i) for i in range(1, 100)
#   # ]
#   file_list = glob.glob(os.path.join(tmp_dir, '*', '*.txt'))
#   return [
#       os.path.join(tmp_dir,
#                    "openwebtext-language-modeling",
#                    "training-monolingual.bpe.shuffled",
#                    "text.en-%05d-of-00100" % i) for i in range(1, 100)
#   ]


# def _dev_data_filenames(tmp_dir):
#   return [os.path.join(tmp_dir,
#                        "openwebtext-language-modeling",
#                        "heldout-monolingual.bpe.shuffled",
#                        "text.en.heldout-00000-of-00050")]


@registry.register_problem
class LanguagemodelOpenWebText(text_problems.Text2SelfProblem):
  """
  A language model on the OpenWebText corpus.
  """

  # @property
  # def approx_vocab_size(self):
  #   # Only for VocabType.SUBWORD
  #   return 50256

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 512,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 32,
    }]

  @property
  def vocab_type(self):
    return text_problems.VocabType.BYTE_PAIR

  def is_generate_per_split(self):
    return True

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    # Get the data filenames and shuffle for train/val split.
    # dataset_filenames = glob.glob(os.path.join(tmp_dir, '*', '*.txt'))
    # random.shuffle(dataset_filenames)
    # training_num = round(len(dataset_filenames) * 0.9)
    # train_data_filenames = dataset_filenames[:training_num]
    # dev_data_filenames = dataset_filenames[training_num:]
    # split_files = {
    #     problem.DatasetSplit.TRAIN: train_data_filenames,
    #     problem.DatasetSplit.EVAL: dev_data_filenames,
    # }
    # Need to make sure the data has been downloaded and prepared!
    # _maybe_download_corpus(tmp_dir)
    # original_vocab = _original_vocab(tmp_dir)

    # Load the byte_pair_encoder.
    # byte_pair_encoder = text_encoder.BytePairEncoder(os.path.join(data_dir, 'encoder.json'), os.path.join(data_dir, 'vocab.bpe'))
    files = train_dev_split(tmp_dir, dataset_split)
    # files = split_files[dataset_split]
    for filepath in files:
      tf.logging.info("filepath = %s", filepath)
      try:
        for line in tf.gfile.Open(filepath):
          # txt = _replace_oov(original_vocab, text_encoder.native_to_unicode(line))
          if line != '\n':
            encoded_txt = line
            yield {"targets": encoded_txt}
      except Exception:
        traceback.print_exc()
        continue
