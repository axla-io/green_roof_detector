#!/bin/bash
#SBATCH --job-name=jax_train
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --gres=gpu:4                 # request 4 GPUs
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load anaconda3/2024.6
module load cudatoolkit/12.8 
conda activate jx-env

# Export CUDA library paths for JAX/XLA to find them
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cudnn/cuda-11.3/8.2.0/lib64:$LD_LIBRARY_PATH

# Optional JAX performance settings
export TF_FORCE_UNIFIED_MEMORY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.3

# Optional: to avoid jaxlib errors with too-new drivers
export JAX_ENABLE_X64=1

# Confirm devices
python -c "import jax; print('JAX devices:', jax.devices())"

# Run your script
python train_jax.py /scratch/gpfs/ll9426/green_roof_data/data/processed