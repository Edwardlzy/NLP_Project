# NLP_Project
Repo for the summer at vector

## Installation
```
conda create -n t2t python=3
conda activate t2t
git clone https://github.com/Edwardlzy/NLP_Project.git
cd tensor2tensor
pip install -e .
pip install tensorflow-gpu
conda install -c conda-forge regex
export PYTHONPATH=`pwd`
```

## Datasets
### OpenWebText
+ Scraped from urls provided in this repo: `https://github.com/jcpeterson/openwebtext`
+ Location: `/scratch/gobi1/datasets/NLP-Corpus/OpenWebText` and `/scratch/hdd001/home/edwardlzy/openwebtext/`

### Data Generation
```
srun --gres=gpu:1 -c 4 --mem=8G -p t4 tensor2tensor/bin/t2t-datagen --data_dir=WHERE_TO_STORE_THE_TFRECORD --tmp_dir=WHERE_TO_FIND_TXT --problem=languagemodel_open_web_text --t2t_usr_dir=tensor2tensor/data_generators/openwebtext/
```
+ Note: the `encoder.json` and `vocab.bpe` should be placed under the `--data_dir`.

### Data Inspection
```
srun --gres=gpu:1 -c 4 --mem=8G -p t4 python tensor2tensor/data_generators/inspect_tfrecord.py --logtostderr --print_targets --bpe_encoder_file=/scratch/hdd001/home/edwardlzy/openwebtext/test_datagen/data/encoder.json --bpe_vocab_file=/scratch/hdd001/home/edwardlzy/openwebtext/test_datagen/data/vocab.bpe --input_filename=/scratch/hdd001/home/edwardlzy/openwebtext/test_datagen/data/languagemodel_open_web_text-dev-00000-of-00001 --print_targets --print_all --byte_pair_encoder
```
+ Sample output:
```
TARGETS:
"But every day demonstrates that this type of care can and must be provided."
"
inputs: []
targets: [1, 1537, 790, 1110, 15687, 326, 428, 2099, 286, 1337, 460, 290, 1276, 307, 2810, 526, 198, 1]
```

### Byte Pair Encoding
+ Intuition: <br />
Leverage the benefits of both word-level and character-level language modeling by interpolating word-level inputs for frequent symbol sequences and char-level inputs for infrequent symbol sequences.
```
Prerequisite: 
  encoder.json: unicode word to index lookup table.
  vocab.bpe: A vocabulary which each word is splitted into 2 unicode strings.

Workflow:
- Plain text
- Split the input to a list of words
- For each char in each word, convert to unicode char
- For each converted unicode word, starting at char (byte) level, iteratively combine bigrams until the new word is not found in vocab.bpe
- Return a list of indices of the words generated from the previous step

Toy Example:
- "Hello world!"
- ["Hello", "world", "!"]
- [b'Hello', b' world', b'!']
- ['Hello', 'Ġworld', '!']
- [15496, 995, 0]
```

## T2T Example Commands
### Language Modeling Training
```
srun --gres=gpu:4 -c 32 -p p100 --mem=64G -w gpu033 t2t-trainer --data_dir=/scratch/hdd001/home/edwardlzy/openwebtext/tfrecords/ --problem=languagemodel_open_web_text --model=transformer --hparams_set=transformer_base --output_dir=/scratch/hdd001/home/edwardlzy/lm_openwebtext_train/ --t2t_usr_dir=/h/edwardlzy/NLP_Project/tensor2tensor/tensor2tensor/data_generators/openwebtext/ --schedule=train
```
### Interactive Decoding
```
srun --gres=gpu:1 -c 16 --mem=16G -p p100 t2t-decoder --data_dir=/scratch/hdd001/home/edwardlzy/openwebtext/tfrecords/ --tmp_dir=/scratch/hdd001/home/edwardlzy/openwebtext/text_data/ --problem=languagemodel_open_web_text --model=transformer --hparams_set=transformer_base --decode_hparams="beam_size=4,alpha=0.6" --decode_interactive --output_dir=/scratch/hdd001/home/edwardlzy/lm_openwebtext_train --t2t_usr_dir=/h/edwardlzy/NLP_Project/tensor2tensor/tensor2tensor/data_generators/openwebtext/
```

### Evaluation
`bash tensor2tensor/eval_lm.sh`
+ Modify the variables at the top
+ Results include the perplexity of the last, avg of the last 5 for every 1k steps, avg of the last 5 for every 5k steps and avg of the last 10 for every 1k steps checkpoints.
+ Look for the results in `$eval_dir/eval_results.txt`

## Logger Example Usage
+ Make sure to set the training hyperparameters in example_master_arguments.txt

### Synchronous Distributed Training
```
python launcher.py --distributed --submit --noautosave --master_address="gpu029:5555" --worker_address="gpu030:5555,gpu053:5555"
```

### Asynchronous Distributed Training
```
python launcher.py --distributed --submit --noautosave --asynchronous --master_address="gpu009:5555,gpu029:5555" --worker_address="gpu030:5555,gpu053:5555"
```

### Checkpoint Averaging
```
C=YOUR_MODEL_DIR
srun --gres=gpu:1 -c 8 --mem=8G -p p100 python tensor2tensor/utils/avg_checkpoints.py --checkpoints="$C/model.ckpt-100000,$C/model.ckpt-95000,$C/model.ckpt-90000,$C/model.ckpt-85000,$C/model.ckpt-80000" --output_path=$C/100k_5k_avg.ckpt
```

#### Note
+ For asynchronous training, use "--schedule=train" to avoid graph mismatch error.


### Experiments
#### Transformer Base on WMT14 EN-DE task
|   | BLEU (uncased) | Iterations | Average | Batch Size | Synchronous |
|---|---|---|---|---|---|
| vanilla | 27.48 | 130K | None | 32k | Yes |
| vanilla_avg_5k | 27.69 | 130k | every 5k steps | 32k | Yes |
| vanilla_avg_1k | 27.71 | 130k | every 1k steps | 32k | Yes |
| vanilla | 27.75 |  100k | None | 48k | Yes |
| vanilla_avg_5k | 27.92 | 100k | every 5k steps | 48k | Yes |
| vanilla_avg_1k | 27.97 | 100k | every 1k steps | 48k | Yes |
| vanilla | 26.41 |  100k | None | 32k | No |

+ All scores are reported on newstest2014.
+ The averaged model from the original Transformer paper has 27.3 bleu score on newstest2014.

#### OpenWebText
| Model | LM1B | WikiText103 | Iterations | Batch Size | Example Length |
|---|---|---|---|---|---|
| GPT2-small | 87.18 | 76.71 | 6016 | 524288 | sentence long |
| GPT2-small | 71.53 | 56.71 | 23218 | 524288 | sentence long |
| GPT2-small | 186.04 | 47.35 | 15625 | 524288 | ~1000 tokens per example |

+ The perplexity is reported on LM1B dev set.

#### LM1B
| Model | Perplexity | Iterations | Encoding | Training Data | Evaluation Data | Optimizer | Batch Size |
|---|---|---|---|---|---|---|---|
| transformer_lm_tpu_0 | 26.77 | 250K | SubwordTextEncoder | LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 26.06 | Avg_1K | SubwordTextEncoder | LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 25.75 | Avg_5K | SubwordTextEncoder | LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 25.85 | Avg_1K_10 | SubwordTextEncoder | LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 25.69 | Avg_1K_20 | SubwordTextEncoder | LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 27.45 | 250K | SubwordTextEncoder | 50% of LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 26.66 | Avg_1K | SubwordTextEncoder | 50% LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 26.46 | Avg_5K | SubwordTextEncoder | 50% LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 26.50 | Avg_1K_10 | SubwordTextEncoder | 50% LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 26.35 | Avg_1K_20 | SubwordTextEncoder | 50% LM1B | LM1B | Adafactor | 16K |
| transformer_lm_tpu_0 | 30.00 | 250K | BytePairEncoder | LM1B | LM1B | Adam | 16K |
| transformer_lm_tpu_0 | 29.23 | Avg_1K | BytePairEncoder | LM1B | LM1B | Adam | 16K |
| transformer_lm_tpu_0 | 29.08 | Avg_5K | BytePairEncoder | LM1B | LM1B | Adam | 16K |
| transformer_lm_tpu_0 | 29.05 | Avg_1K_10 | BytePairEncoder | LM1B | LM1B | Adam | 16K |
| transformer_lm_tpu_0 | 28.96 | Avg_1K_20 | BytePairEncoder | LM1B | LM1B | Adam | 16K |
| transformer_lm_tpu_0 | 1217.72 | Avg_1K_20 | BytePairEncoder | LM1B | WikiText103 | Adam | 16K |

#### LM1B Deprecated
| Model | Perplexity | Iterations | Encoding | Training Data | Optimizer | Batch Size |
|---|---|---|---|---|---|---|
| transformer_base | 42.50 | 250K | SubwordTextEncoder | LM1B Training Set | Adam | 4K |
| transformer_base | 140.70 | 250K | BytePairEncoder | OpenWebText | Adam | 4K |
| transformer_base | 43.68 | 250K | BytePairEncoder | LM1B Training Set| Adam | 4K |
| transformer_big | 152.70  | 250K | BytePairEncoder | OpenWebText| Adam | 4K |
| transformer_big | 44.08 | 250K | SubwordTextEncoder | LM1B Training Set | Adam | 4K |
| transformer_lm_tpu_0 | 31.48 | 250K | SubwordTextEncoder | LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 30.18 | Avg_1K | SubwordTextEncoder | LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 29.75 | Avg_5K | SubwordTextEncoder | LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 29.86 | Avg_1K_10 | SubwordTextEncoder | LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 31.72 | 250K | SubwordTextEncoder | 50% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 30.50 | Avg_1K | SubwordTextEncoder | 50% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 29.96 | Avg_5K | SubwordTextEncoder | 50% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 30.12 | Avg_1K_10 | SubwordTextEncoder | 50% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 32.41 | 250K | SubwordTextEncoder | 25% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 31.09 | Avg_1K | SubwordTextEncoder | 25% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 30.56 | Avg_5K | SubwordTextEncoder | 25% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 30.72 | Avg_1K_10 | SubwordTextEncoder | 25% of LM1B Training Set | Adafactor | 4K |
| transformer_lm_tpu_0 | 40.44 | 250K | BytePairEncoder | LM1B Training Set | Adam | 4K |
| transformer_lm_tpu_0 | 38.91 | Avg_1K | BytePairEncoder | LM1B Training Set | Adam | 4K |
| transformer_lm_tpu_0 | 38.81 | Avg_5K | BytePairEncoder | LM1B Training Set | Adam | 4K |
| transformer_lm_tpu_0 | 41.05 | 250K | BytePairEncoder | LM1B Training Set | Lookahead_Adam | 4K |
| transformer_lm_tpu_0 | 40.89 | 250K | BytePairEncoder | LM1B Training Set | Lookahead_Adam | 4K |

+ Perplexity is reported on LM1B dev set.
