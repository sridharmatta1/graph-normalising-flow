#!/bin/bash
#SBATCH --job-name=sridhar
#SBATCH --output=output.log
#SBATCH --error=error.err
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

source activate SRP_2026_Graph
srun python <your_script>.py

