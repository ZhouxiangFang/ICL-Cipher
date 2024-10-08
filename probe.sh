#!/bin/bash

#SBATCH -A mdredze80_gpu
#SBATCH --partition ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=36GB
#SBATCH --qos=qos_gpu
#SBATCH --job-name="icl"
#SBATCH --output="/home/zfang27/icl/log/log-%j.txt" # Path to store logs

# danielk_gpu;mdredze80_gpu; mdredze1_gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONV_RSH=ssh
export HF_HOME=/data/danielk/zfang27/cache

dataset=$1
top=$2
fewshot=$3
sub=$4

srun python -m probe \
        --dataset ${dataset}\
        --top ${top} \
        --fewshot ${fewshot} \
        --sub ${sub} \
        # --reverse \
        
        
        
        
        