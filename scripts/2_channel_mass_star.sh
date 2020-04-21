#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --mem=120G  # memory in Mb
#SBATCH --partition=mlgpu
#SBATCH -o outfile-2_channel_mass_star  # send stdout to outfile
#SBATCH -e errfile-2_channel_mass_star  # send stderr to errfile

cd ~/GalaxyEnvironmentAnalysis
source env/bin/activate
python gea.py --debug train config/1.1-2_channel_mass_star.yml --new-model
