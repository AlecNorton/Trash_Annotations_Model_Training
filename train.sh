#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16g
#SBATCH -J "Training Segmentation"
#SBATCH --gres=gpu:1
#SBATCH -C "A100|V100"


module load python/3.12.7

source ~/MQP/venv/bin/activate


python test.py -e 10 -b 10
echo "Deactivate"
deactivate

echo "Training Segmentation Job With LR - 0.001 and Weight_decay - 0.0001"
