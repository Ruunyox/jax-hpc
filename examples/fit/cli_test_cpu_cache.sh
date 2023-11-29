#! /bin/bash
#SBATCH -J cli_test_cpu_cache
#SBATCH -o ./log/cli_test_cpu_cache.out
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load anaconda3/2023.09
conda activate base 

export JAX_PLATFORM_NAME=cpu
export PYTHONUNBUFFERED=on

jaxhpc --config config_local_cache.yaml
