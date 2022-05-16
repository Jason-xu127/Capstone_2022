#!/usr/bin/bash
#
#SBATCH --job-name=myTrainTask2GPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=4820v3
#SBATCH --output=log/log_train_fix_%A_%a.out
#SBATCH --error=log/log_train_fix_%A_%a.err
#SBATCH --time=2-00:00:00


num_gpus=1

module purge

export PATH=$HOME/anaconda3/bin:$PATH

source activate dstc10_task1

python run_generation.py