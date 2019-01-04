#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:10:00
#SBATCH --mem=1GB
#SBATCH --job-name=gpu_svm_sample
##SBATCH --mail-type=END
##SBATCH --mail-user=drr342@nyu.edu
#SBATCH --output=gpu_svm_sample%j.out

module purge
module load cuda/9.2.88

./runSample.sh
