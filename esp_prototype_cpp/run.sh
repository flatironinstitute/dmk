#!/bin/bash
#SBATCH --job-name=esp
#SBATCH --output=esp_output.log
#SBATCH --error=esp_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

export OMP_NUM_THREADS=8
./build/esp