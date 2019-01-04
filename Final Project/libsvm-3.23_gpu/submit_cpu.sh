#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=cpu_svm
##SBATCH --mail-type=END
##SBATCH --mail-user=drr342@nyu.edu
#SBATCH --output=cpu_svm_%j.out

module purge
module load cuda/9.2.88

BASE_PATH=/scratch/drr342/mnist
MODEL=$BASE_PATH/mnist.scale.model

echo "Scaling MNIST 1k on CPU"
time ./svm-scale -l 0 -u 1 $BASE_PATH/mnist1k
echo ""
echo "Scaling MNIST 10k on CPU"
time ./svm-scale -l 0 -u 1 $BASE_PATH/mnist10k
echo ""
echo "Scaling MNIST 100k on CPU"
time ./svm-scale -l 0 -u 1 $BASE_PATH/mnist100k
echo ""
echo "Scaling MNIST 1M on CPU"
time ./svm-scale -l 0 -u 1 $BASE_PATH/mnist1m
echo ""
echo "Scaling MNIST 8M on CPU"
time ./svm-scale -l 0 -u 1 $BASE_PATH/mnist8m
echo ""

echo "Predicting MNIST 1k on CPU"
time ./svm-predict $BASE_PATH/mnist1k.scale.cpu $MODEL
echo ""
echo "Predicting MNIST 10k on CPU"
time ./svm-predict $BASE_PATH/mnist10k.scale.cpu $MODEL
echo ""
echo "Predicting MNIST 100k on CPU"
time ./svm-predict $BASE_PATH/mnist100k.scale.cpu $MODEL
echo ""
echo "Predicting MNIST 1M on CPU"
time ./svm-predict $BASE_PATH/mnist1m.scale.cpu $MODEL
echo ""
echo "Predicting MNIST 8M on CPU"
time ./svm-predict $BASE_PATH/mnist8m.scale.cpu $MODEL
echo ""
