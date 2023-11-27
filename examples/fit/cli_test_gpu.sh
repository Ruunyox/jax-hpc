#! /bin/bash
#SBATCH -J cli_test_gpu
#SBATCH -o cli_test_gpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --reservation=a100_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load sw.a100
module load nvhpc/23.1
module load cuda/11.8
module load anaconda3/2023.09
conda activate base

export JAX_PLATFORM_NAME=gpu
export PYTHONUNBUFFERED=on

jaxhpc --config config_local_gpu.yaml
