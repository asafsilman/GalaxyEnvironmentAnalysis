import os
from subprocess import check_output, Popen, PIPE, STDOUT

SLURM_SCRIPT_TEMPLATE = \
"""#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --mem={1}G  # memory in Mb
#SBATCH --partition={2}
#SBATCH -o outfile-{0}  # send stdout to outfile
#SBATCH -e errfile-{0}  # send stderr to errfile
#SBATCH -d afterok:{5}
#SBATCH -J gea-training

cd {3}
source env/bin/activate
{4}
"""

def get_training_command(model_name):
    return f"python gea.py --debug train config/{model_name}.yml --new-model"

def jobs_running_on_partition(partition):
    command = [
        "squeue",
        "--noheader",
        f"--partition={partition}",
        "--format=\"%i\""
    ]
    output = check_output(command).decode().split("\n")
    return list(
        map(
            lambda x: x.replace("\"", ""),
            filter(
                lambda x: x!="", output
            )
        )
    )

def get_slurm_script(model_name, mem_limit_gb=40, partition="mlgpu", project_directory="~/GalaxyEnvironmentAnalysis"):
    training_command = get_training_command(model_name)
    running_jobs = ",".join(jobs_running_on_partition(partition) or [0])
    
    return SLURM_SCRIPT_TEMPLATE.format(
        model_name,
        mem_limit_gb,
        partition,
        project_directory,
        training_command,
        running_jobs
    )

def submit_job_slurm(model_name, **kwargs):
    script = get_slurm_script(model_name, **kwargs)
    
    p = Popen(["sbatch"], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    output = p.communicate(input=script.encode())[0]

    return_code = p.wait()
    if return_code:
        return True
    else:
        print(output.decode())
        return False
