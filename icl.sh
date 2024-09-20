#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 03:00:00
#SBATCH --mem-per-cpu=24GB
#SBATCH --qos=qos_gpu
#SBATCH --job-name="icl"
#SBATCH --output="/home/zfang27/icl/log/log-%j.txt" # Path to store logs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CONV_RSH=ssh
export HF_HOME=/data/danielk/zfang27/cache

dataset=$1 
model=$2
zipfian=$3
fewshot=$4
shuffle_rate=$5
run=$6

srun python -m icl \
        --dataset ${dataset}\
        --model ${model} \
        --zipfian ${zipfian} \
        --fewshot ${fewshot} \
        --shuffle_rate ${shuffle_rate} \
        --run ${run} \
        # --prefix \
        # --OS \
        # --nonbi \
        
        
        
        
        