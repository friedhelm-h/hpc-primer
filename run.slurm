#!/bin/bash

#SBATCH -o ./log/%j.%N.out
#SBATCH -D ./
#SBATCH -J my-test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:05:00 
#SBATCH --gres=gpu:v100s:1

#SBATCH --partition=ex_scioi_gpu

pwd; hostname; date

export https_proxy=http://frontend01:3128/
export http_proxy=http://frontend01:3128/

module load singularity/3.7.0

singularity run --pwd /wd --bind /beegfs/scratch/fhamann:/wd images/my-pytorch_latest.sif python3 scripts/feedforward.py