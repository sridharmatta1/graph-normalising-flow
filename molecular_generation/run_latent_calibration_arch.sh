#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# Per-dimension latent calibration for one architecture variant (A-E) of the
# N-conditioned flow, from its best-validation checkpoint, on both the
# training and the held-out validation embeddings. Same report as
# run_latent_calibration_diagnostic.sh; the key columns for the new variants:
#   nfloor  how many of the 64 prior sigmas sit at the 0.1 floor
#   zRMS    1.00 = prior spread matches the latent
# Reads an existing checkpoint, writes one new JSON; no training.
#
#   sbatch --job-name=calib_arch_A --export=ALL,ARCH=A \
#       molecular_generation/run_latent_calibration_arch.sh
# ============================================================
ARCH=${ARCH:-A}

case $ARCH in
    A) ARCH_FLAGS="--coupling_arch attn_film" ;;
    B) ARCH_FLAGS="--coupling_arch deep_film" ;;
    C) ARCH_FLAGS="--coupling_arch interaction_film" ;;
    D) ARCH_FLAGS="--coupling_arch film --channel_permutation" ;;
    E) ARCH_FLAGS="--coupling_arch spline_film" ;;
    *) echo "ARCH must be one of A B C D E"; exit 1 ;;
esac

PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

cd $WORKDIR
mkdir -p logs molecular_generation/results

echo "##### Latent calibration, architecture $ARCH ($ARCH_FLAGS) #####"
srun $PYTHON -u molecular_generation/inspect_latent_calibration.py \
    --model_kind n_conditioned \
    $ARCH_FLAGS \
    --flow_checkpoint molecular_generation/test_runs/flow_arch_$ARCH/best/best \
    --save_summary_json molecular_generation/results/latent_calibration_arch_$ARCH.json

echo "ALL DONE: $(date)"
