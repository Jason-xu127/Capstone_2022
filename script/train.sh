#!/usr/bin/bash
#
#SBATCH --job-name=myTrainTask2GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=4820v3
#SBATCH --output=log/log_train_fix_%A_%a.out
#SBATCH --error=log/log_train_fix_%A_%a.err
#SBATCH --time=2-00:00:00


num_gpus=1

module purge

export PATH=$HOME/anaconda3/bin:$PATH

source activate dstc10_task1

python run_summarization.py \
    --model_name_or_path facebook/bart-large-cnn \
    --tokenizer_name facebook/bart-large-cnn-local \
    --do_train \
    --do_eval \
    --train_file train_v2.json\
    --validation_file valid_v2.json\
    --output_dir tmp/tst-summarization \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=4 \