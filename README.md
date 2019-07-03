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
python launcher.py --distributed --submit --noautosave --master_address="gpu009:5555,gpu029:5555" --worker_address="gpu030:5555,gpu053:5555"
```

#### Note
+ For asynchronous training, use "--schedule=train" to avoid graph mismatch error.


### Experiments
#### Transformer Base on WMT14 EN-DE task
|   | BLEU (cased) | Iterations | Average | Batch Size | Synchronous |
|---|---|---|---|---|---|
| vanilla | 27.48 | 130K | None | 8k | Yes |
| vanilla_avg_5k | 27.69 | 130k | every 5k steps | 8k | Yes |
| vanilla_avg_1k | 27.71 | 130k | every 1k steps | 8k | Yes |
| vanilla | 27.75 |  100k | None | 12k | Yes |
| vanilla_avg_5k | 27.92 | 100k | every 5k steps | 12k | Yes |
| vanilla_avg_1k | 27.97 | 100k | every 1k steps | 12k | Yes |

+ All scores are reported on newstest2014.
+ The averaged model from the original Transformer paper has 27.3 bleu score on newstest2014.
