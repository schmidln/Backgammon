#!/bin/bash
#SBATCH --job-name=agent1_gpu_bill
#SBATCH --account=bfxm-delta-gpu   # <--- UPDATED: Matches your output
#SBATCH --partition=gpuA100x4      # <--- UPDATED: Must use GPU partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1          # <--- REQUIRED: Ticket to enter
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16         # Getting 16 powerful AMD Milan cores
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=training_log.txt

# 1. Load Environment
module load python
source /scratch/bfxm/tcliftoniv1/bg_env/bin/activate

# 2. Go to directory
cd /scratch/bfxm/tcliftoniv1/backgammon

# 3. Optimize Parallelism
# We have 16 physical cores now, so we tune for that
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16

# 4. Run!
python td_gammon_agent.py
