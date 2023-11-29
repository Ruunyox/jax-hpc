#! /bin/bash
#SBATCH -J cli_test_gpu_cache
#SBATCH -o ./log/cli_test_gpu_cache.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --reservation=a100_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load cuda/11.8
module load anaconda3/2023.09
conda activate base

export JAX_PLATFORM_NAME=gpu
export PYTHONUNBUFFERED=on
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/sw/compiler/cuda/11.8/a100/install

jaxhpc --config config_local_gpu_cache.yaml
