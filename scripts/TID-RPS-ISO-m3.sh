#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --mem=40G  # memory in Mb
#SBATCH --partition=mlgpu
#SBATCH -o outfile-TID-RPS-ISO-m3  # send stdout to outfile
#SBATCH -e errfile-TID-RPS-ISO-m3  # send stderr to errfile

cd ~/GalaxyEnvironmentAnalysis
source env/bin/activate
python gea.py --debug train config/1.0-TID-RPS-ISO-m3.yml --new-model
