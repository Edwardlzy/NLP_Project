#!/bin/bash

eval_dir=/scratch/hdd001/home/edwardlzy/multi_gpu/lm1b_transformer_lm_tpu_0 #lm1b_transformer_lm_tpu_1
data_dir=/scratch/hdd001/home/edwardlzy/lm1b_data/
problem=languagemodel_lm1b32k
hparams_set=transformer_lm_tpu_0
usr_dir=/h/edwardlzy/NLP_Project/tensor2tensor/tensor2tensor/data_generators/openwebtext/
nlp_repo=/h/edwardlzy/NLP_Project/

# Eval the last checkpoint first
echo "Evaluating the last checkpoint..."
mv $eval_dir/checkpoint $eval_dir/checkpoint_bk
echo 'model_checkpoint_path: "model.ckpt-250000"' > $eval_dir/checkpoint
# Run evaluation
srun --gres=gpu:1 -c 8 --mem=16G -p max12hours t2t-eval --data_dir=$data_dir --problem=$problem --model=transformer --hparams_set=$hparams_set --output_dir=$eval_dir --t2t_usr_dir=$usr_dir --hparams="batch_size=1024" --eval_steps=1000 --eval_use_test_set --eval_timeout_mins=0 &> $eval_dir/last_ckpt_eval.txt

# Read loss from the eval result
loss=`cat $eval_dir/last_ckpt_eval.txt | grep -E -o 'loss = [0-9].[0-9]+' | grep -E -o '[0-9].[0-9]+'`
perplexity=`python -c "import math;print(math.exp($loss))"`
echo "For the last checkpoint, the perplexity = $perplexity"
echo "For the last checkpoint, the perplexity = $perplexity" >> $eval_dir/eval_results.txt

# Eval the last 5 checkpoints
echo "Averaging last 5 checkpoints for every 1k steps..."
srun --gres=gpu:1 -c 8 --mem=8G -p max12hours python $nlp_repo/tensor2tensor/tensor2tensor/utils/avg_checkpoints.py --checkpoints="$eval_dir/model.ckpt-250000,$eval_dir/model.ckpt-249000,$eval_dir/model.ckpt-248000,$eval_dir/model.ckpt-247000,$eval_dir/model.ckpt-246000" --output_path=$eval_dir/1k_avg.ckpt

echo "Evaluating the average of the last 5 checkpoints for every 1k steps..."
srun --gres=gpu:1 -c 8 --mem=16G -p max12hours t2t-eval --data_dir=$data_dir --problem=$problem --model=transformer --hparams_set=$hparams_set --output_dir=$eval_dir --t2t_usr_dir=$usr_dir --hparams="batch_size=1024" --eval_steps=1000 --eval_use_test_set --eval_timeout_mins=0 &> $eval_dir/1k_avg_eval.txt
loss=`cat $eval_dir/1k_avg_eval.txt | grep -E -o 'loss = [0-9].[0-9]+' | grep -E -o '[0-9].[0-9]+'`
perplexity=`python -c "import math;print(math.exp($loss))"`
echo "For the average of every 1k checkpoints, the perplexity = $perplexity"
echo "For the average of every 1k checkpoints, the perplexity = $perplexity" >> $eval_dir/eval_results.txt

# Eval the last 5 checkpoints for every 5k steps
echo "Averaging last 5 checkpoints for every 5k steps..."
srun --gres=gpu:1 -c 8 --mem=8G -p max12hours python $nlp_repo/tensor2tensor/tensor2tensor/utils/avg_checkpoints.py --checkpoints="$eval_dir/model.ckpt-250000,$eval_dir/model.ckpt-245000,$eval_dir/model.ckpt-240000,$eval_dir/model.ckpt-235000,$eval_dir/model.ckpt-231000" --output_path=$eval_dir/5k_avg.ckpt

echo "Evaluating the average of the last 5 checkpoints for every 5k steps..."
srun --gres=gpu:1 -c 8 --mem=16G -p max12hours t2t-eval --data_dir=$data_dir --problem=$problem --model=transformer --hparams_set=$hparams_set --output_dir=$eval_dir --t2t_usr_dir=$usr_dir --hparams="batch_size=1024" --eval_steps=1000 --eval_use_test_set --eval_timeout_mins=0 &> $eval_dir/5k_avg_eval.txt
loss=`cat $eval_dir/5k_avg_eval.txt | grep -E -o 'loss = [0-9].[0-9]+' | grep -E -o '[0-9].[0-9]+'`
perplexity=`python -c "import math;print(math.exp($loss))"`
echo "For the average of every 5k checkpoints, the perplexity = $perplexity"
echo "For the average of every 5k checkpoints, the perplexity = $perplexity" >> $eval_dir/eval_results.txt

# Eval the average of the last 10 checkpoints
echo "Averaging last 10 checkpoints for every 1k steps..."
srun --gres=gpu:1 -c 8 --mem=8G -p max12hours python $nlp_repo/tensor2tensor/tensor2tensor/utils/avg_checkpoints.py --checkpoints="$eval_dir/model.ckpt-250000,$eval_dir/model.ckpt-249000,$eval_dir/model.ckpt-248000,$eval_dir/model.ckpt-247000,$eval_dir/model.ckpt-246000,$eval_dir/model.ckpt-245000,$eval_dir/model.ckpt-244000,$eval_dir/model.ckpt-243000,$eval_dir/model.ckpt-242000,$eval_dir/model.ckpt-241000" --output_path=$eval_dir/1k_10_avg.ckpt

echo "Evaluating the average of the last 10 checkpoints..."
srun --gres=gpu:1 -c 8 --mem=16G -p max12hours t2t-eval --data_dir=$data_dir --problem=$problem --model=transformer --hparams_set=$hparams_set --output_dir=$eval_dir --t2t_usr_dir=$usr_dir --hparams="batch_size=1024" --eval_steps=1000 --eval_use_test_set --eval_timeout_mins=0 &> $eval_dir/1k_10_avg_eval.txt
loss=`cat $eval_dir/1k_10_avg_eval.txt | grep -E -o 'loss = [0-9].[0-9]+' | grep -E -o '[0-9].[0-9]+'`
perplexity=`python -c "import math;print(math.exp($loss))"`
echo "For the average 10 every 1k checkpoints, the perplexity = $perplexity"
echo "For the average 10 every 1k checkpoints, the perplexity = $perplexity" >> $eval_dir/eval_results.txt

# Eval the average of the last 20 checkpoints
echo "Averaging last 20 checkpoints for every 1k steps..."
srun --gres=gpu:1 -c 8 --mem=8G -p max12hours python $nlp_repo/tensor2tensor/tensor2tensor/utils/avg_checkpoints.py --checkpoints="$eval_dir/model.ckpt-250000,$eval_dir/model.ckpt-249000,$eval_dir/model.ckpt-248000,$eval_dir/model.ckpt-247000,$eval_dir/model.ckpt-246000,$eval_dir/model.ckpt-245000,$eval_dir/model.ckpt-244000,$eval_dir/model.ckpt-243000,$eval_dir/model.ckpt-242000,$eval_dir/model.ckpt-241000,$eval_dir/model.ckpt-240000,$eval_dir/model.ckpt-239000,$eval_dir/model.ckpt-238000,$eval_dir/model.ckpt-237000,$eval_dir/model.ckpt-236000,$eval_dir/model.ckpt-235000,$eval_dir/model.ckpt-234000,$eval_dir/model.ckpt-233000,$eval_dir/model.ckpt-232000,$eval_dir/model.ckpt-231000" --output_path=$eval_dir/1k_20_avg.ckpt

echo "Evaluating the average of the last 20 checkpoints..."
srun --gres=gpu:1 -c 8 --mem=16G -p max12hours t2t-eval --data_dir=$data_dir --problem=$problem --model=transformer --hparams_set=$hparams_set --output_dir=$eval_dir --t2t_usr_dir=$usr_dir --hparams="batch_size=1024" --eval_steps=1000 --eval_use_test_set --eval_timeout_mins=0 &> $eval_dir/1k_20_avg_eval.txt
loss=`cat $eval_dir/1k_20_avg_eval.txt | grep -E -o 'loss = [0-9].[0-9]+' | grep -E -o '[0-9].[0-9]+'`
perplexity=`python -c "import math;print(math.exp($loss))"`
echo "For the average 20 every 1k checkpoints, the perplexity = $perplexity"
echo "For the average 20 every 1k checkpoints, the perplexity = $perplexity" >> $eval_dir/eval_results.txt

echo "results saved in $eval_dir/eval_results.txt"
echo "done"
