LibSVM: A parallelized implementation using GPUs

Daniel Rivera Ruiz
drr342@nyu.edu
New York University

The code included in libsvm-3.23_gpu provides all the functionalities of the original LibSVM library (scale, train, predict), plus the parallelized counterparts.

IMPORTANT:
All the following directives assume access to the Prince HPC Cluster. If running elsewhere, some modifications might be necessary (i.e. arch flag for the nvcc compiler according to available GPU) and some functionalities might not be available (i.e. slurm batch scheduling).

To compile the code, navigate to libsvm-3.23_gpu and run the Makefile. (Make sure to load the cuda/9.2.88 module before compilation)

The three executables svm-scale, svm-train and svm-predict will function EXACTLY as the ones in the original LibSVM (refer to the libSVM README file).

The three executables svm-scale-gpu, svm-train-gpu and svm-predict-gpu are the counterparts that execute in the GPU. The way of executing them is the same (refer to the libSVM README file), although the output generated is a little different (considering that the parts of the code that run in the GPU do not have access to stdout).

Running any of the executables without arguments will also provide helpful information on how to run them (the same that is available on the libSVM README)

To recreate the results presented in the final report, submit the following slurm jobs:
    1) sbatch submit_cpu.sh
    2) sbatch submit_gpu.sh
    3) sbatch submit_nvprof.sh
Uncomment the #SBATCH directive for email notification to receive an email when the job has completed.

To test the functionalities of the library in the small dataset sampleData that is provided, submit the slurm job:
    1) sbatch submit_sample.sh

Alternatively, run it in an interactive terminal by doing:
    1) srun -t0:10:00 --mem=4GB --gres=gpu:v100:1 --pty /bin/bash
    2) module purge
    3) module load cuda/9.2.88
    4) ./runSample.sh

All the data from the MNIST dataset in LibSVM format used in the project experiments is available at /scratch/drr342/mnist
