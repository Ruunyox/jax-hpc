#! /bin/bash
#SBATCH -J cli_test_gpu
#SBATCH -o cli_test_gpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load anaconda3/2023.09
conda activate base

export JAX_PLATFORM_NAME=gpu

python3 -u test_gpu.py
