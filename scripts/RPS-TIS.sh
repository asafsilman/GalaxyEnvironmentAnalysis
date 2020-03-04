#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 10      # cores requested
#SBATCH --mem=180G  # memory in Mb
#SBATCH --partition=mlgpu
#SBATCH -o outfile-RPS-RID  # send stdout to outfile
#SBATCH -e errfile-RPS-RID  # send stderr to errfile

cd ~/GalaxyEnvironmentAnalysis
source env/bin/activate
python gea.py --debug data-prep-split config/1.0-RPS-TID.yml
python gea.py --debug train config/1.0-RPS-TID.yml --new-model
