#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=nvprof_svm
##SBATCH --mail-type=END
##SBATCH --mail-user=drr342@nyu.edu
#SBATCH --output=nvprof_svm_%j.out

module purge
module load cuda/9.2.88

BASE_PATH=/scratch/drr342/mnist
MODEL=$BASE_PATH/mnist.scale.model

echo "NVPROF Scaling MNIST 1k on GPU"
nvprof ./svm-scale-gpu -l 0 -u 1 $BASE_PATH/mnist1k
echo ""
echo "NVPROF Scaling MNIST 10k on GPU"
nvprof ./svm-scale-gpu -l 0 -u 1 $BASE_PATH/mnist10k
echo ""
echo "NVPROF Scaling MNIST 100k on GPU"
nvprof ./svm-scale-gpu -l 0 -u 1 $BASE_PATH/mnist100k
echo ""
echo "NVPROF Scaling MNIST 1M on GPU"
nvprof ./svm-scale-gpu -l 0 -u 1 $BASE_PATH/mnist1m
echo ""
echo "NVPROF Scaling MNIST 8M on GPU"
nvprof ./svm-scale-gpu -l 0 -u 1 $BASE_PATH/mnist8m
echo ""

echo "NVPROF Predicting MNIST 1k on GPU"
nvprof ./svm-predict-gpu $BASE_PATH/mnist1k.scale.gpu $MODEL
echo ""
echo "NVPROF Predicting MNIST 10k on GPU"
nvprof ./svm-predict-gpu $BASE_PATH/mnist10k.scale.gpu $MODEL
echo ""
echo "NVPROF Predicting MNIST 100k on GPU"
nvprof ./svm-predict-gpu $BASE_PATH/mnist100k.scale.gpu $MODEL
echo ""
echo "NVPROF Predicting MNIST 1M on GPU"
nvprof ./svm-predict-gpu $BASE_PATH/mnist1m.scale.gpu $MODEL
echo ""
echo "NVPROF Predicting MNIST 8M on GPU"
nvprof ./svm-predict-gpu $BASE_PATH/mnist8m.scale.gpu $MODEL
echo ""
