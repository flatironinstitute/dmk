#!/bin/bash
#SBATCH --job-name=test_poly
#SBATCH --output=test_poly_output.log
#SBATCH --error=test_poly_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00

./build/test_poly