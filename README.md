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
+ Size: 152G

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
