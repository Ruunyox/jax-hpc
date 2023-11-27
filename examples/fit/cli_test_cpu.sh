#! /bin/bash
#SBATCH -J cli_test_cpu
#SBATCH -o cli_test_cpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

conda activate YOUR_CONDA_ENV

export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=on

jaxhpc --config config_local.yaml
