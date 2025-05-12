#!/bin/bash
#SBATCH --job-name=JAXcpu         # Job name
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --time=3:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all        # send email when job begins
#SBATCH --mail-user=ll9426@princeton.edu
#SBATCH --output=job_JAXcpu_output_%j.txt   # Output file (%j will be replaced by job ID)
#SBATCH --error=job_JAXcpu_error_%j.txt     # Error file (%j will be replaced by job ID)


module purge
module load anaconda3/2024.6
conda activate jx-env

# Run your script
python train_jax_cpu.py /scratch/gpfs/ll9426/green_roof_data/data/processed