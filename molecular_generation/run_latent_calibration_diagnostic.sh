#!/bin/bash
#SBATCH --job-name=GNF_latent_calibration
#SBATCH --output=logs/latent_calibration_output.log
#SBATCH --error=logs/latent_calibration_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Per-dimension latent calibration, for three trained flows, on both
# the training embeddings and the held-out val embeddings:
#   1. N-conditioned, early-stopped best (200k steps)
#   2. N-conditioned, original fixed 100k-step run
#   3. Unconditioned, 100k steps (fixed N(0, I) prior) -- the reference
#
# Decides whether the conditioned model's prior spread is actually
# miscalibrated (and so whether pinning/regularizing sigma is the fix),
# or whether the latent is calibrated but heavy-tailed. Reads only;
# restores existing checkpoints and writes new JSON files.
#
# Needs the val embeddings from run_molecular_flow_n_conditioned_
# earlystop.sh (molecular_generation/data/val_embeddings).
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
RUNS=molecular_generation/test_runs

mkdir -p $WORKDIR/logs $WORKDIR/molecular_generation/results

cd $WORKDIR

echo "##### 1/3 N-conditioned, early-stopped best #####"
srun $PYTHON -u molecular_generation/inspect_latent_calibration.py \
    --model_kind n_conditioned \
    --flow_checkpoint $RUNS/flow_n_conditioned_earlystop/best/best \
    --save_summary_json molecular_generation/results/latent_calibration_n_cond_earlystop.json

echo "##### 2/3 N-conditioned, original 100k #####"
srun $PYTHON -u molecular_generation/inspect_latent_calibration.py \
    --model_kind n_conditioned \
    --flow_checkpoint $RUNS/flow_n_conditioned/checkpoints-100001 \
    --save_summary_json molecular_generation/results/latent_calibration_n_cond_100k.json

echo "##### 3/3 Unconditioned, 100k (fixed N(0,I) prior) #####"
srun $PYTHON -u molecular_generation/inspect_latent_calibration.py \
    --model_kind unconditioned \
    --flow_checkpoint $RUNS/flow_full/checkpoints-100001 \
    --save_summary_json molecular_generation/results/latent_calibration_uncond_100k.json

echo "ALL DONE: $(date)"
