"""Data generators for OpenWebText data-set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
from six.moves import range  # pylint: disable=redefined-builtin
import glob
import random

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf

split_files = None

def train_dev_split(tmp_dir, split, ratio=0.9):
  """Split the data into training and validation set."""
  global split_files
  if not split_files:
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
    byte_pair_encoder = text_encoder.BytePairEncoder(os.path.join(data_dir, 'encoder.json'), os.path.join(data_dir, 'vocab.bpe'))
    files = train_dev_split(tmp_dir, dataset_split)
    # files = split_files[dataset_split]
    for filepath in files:
      tf.logging.info("filepath = %s", filepath)
      for line in tf.gfile.Open(filepath):
        # txt = _replace_oov(original_vocab, text_encoder.native_to_unicode(line))
        if line != '\n':
          #encoded_txt = byte_pair_encoder.encode(line)
          encoded_txt = line
          yield {"targets": encoded_txt}
