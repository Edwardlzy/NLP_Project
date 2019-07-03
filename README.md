# NLP_Project
Repo for the summer at vector

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
| vanilla | 27.48 | 130K | None | 8k | Yes |
| vanilla_avg_5k | 27.69 | 130k | every 5k steps | 8k | Yes |
| vanilla_avg_1k | 27.71 | 130k | every 1k steps | 8k | Yes |
| vanilla | 27.75 |  100k | None | 12k | Yes |
| vanilla_avg_5k | 27.92 | 100k | every 5k steps | 12k | Yes |
| vanilla_avg_1k | 27.97 | 100k | every 1k steps | 12k | Yes |

+ All scores are reported on newstest2014.
+ The averaged model from the original Transformer paper has 27.3 bleu score on newstest2014.
