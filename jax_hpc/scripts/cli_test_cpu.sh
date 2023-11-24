#! /bin/bash
#SBATCH -J cli_test_cpu
#SBATCH -o cli_test_cpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load anaconda3/2023.09
conda activate base

export JAX_PLATFORM_NAME=cpu

python3 -u test.py
