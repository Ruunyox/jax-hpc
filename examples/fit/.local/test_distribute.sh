#! /bin/bash
#SBATCH -J test_distribute
#SBATCH -o ./logs/test_distribute.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --reservation=a100_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load sw.a100
module load cuda/11.8
module load anaconda3/2023.09
module load nvhpc-hpcx/23.1

conda activate base

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=GRAPH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/sw/compiler/cuda/11.8/a100/install

python distributed_test.py
