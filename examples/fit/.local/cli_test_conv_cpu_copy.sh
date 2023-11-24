#! /bin/bash
#SBATCH -J cli_test_cpu_conv
#SBATCH -o ./logs/cli_test_cpu_conv.out
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load anaconda3/2023.09
conda activate tf_test

tfhpc --config config_conv_local.yaml
