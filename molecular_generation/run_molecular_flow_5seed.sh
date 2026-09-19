#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_5seed
#SBATCH --output=logs/molecular_flow_5seed_output_%a.log
#SBATCH --error=logs/molecular_flow_5seed_error_%a.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --array=1-5

# ============================================================
# 5-seed evaluation of the UNCONDITIONED molecular flow, for a
# statistically defensible mean +/- std comparison against
# MoFlow's own reported protocol (Table 2 reports mean +/- std
# over 5 runs), instead of the single-seed numbers used so far.
#
# Cheaper than the ego/community 5-seed array: Phase 2 (frozen
# encoder/decoder) and Phase 3 (extracted embeddings) do NOT
# depend on this seed and are reused unchanged across all 5 --
# only Phase 5 (the flow itself) gets retrained per seed, then
# Phase 6 (generation) runs once against each seed's resulting
# checkpoint. No re-running Phase 1-3 five times.
#
# Each seed:
#   1. Trains train_molecular_flow.py with --random_seed=$SEED,
#      identical hyperparameters to the original single-seed run
#      (run_molecular_flow_full.sh) otherwise.
#   2. Generates 10,000 molecules from that seed's checkpoint
#      (matching the sample size already established as
#      statistically meaningful for this project -- n=100 vs
#      n=10,000 gave very different uniqueness readings).
#
# After all 5 array tasks finish, run
# aggregate_seed_results.py to compute mean +/- std across them.
# ============================================================
SEED=$SLURM_ARRAY_TASK_ID
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_LOGDIR=molecular_generation/test_runs/flow_full_seed_$SEED

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular GNF flow (unconditioned): seed $SEED"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow.py \
    --train_data_dir molecular_generation/data \
    --logdir $FLOW_LOGDIR \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --max_log_scale 0.25 \
    --num_train_iters 100000 \
    --train_batch_size 32 \
    --random_seed $SEED \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_full_qm9_seed${SEED}"

echo "=============================================="
echo "Flow training complete (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

echo "=============================================="
echo "Generating 10,000 molecules (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/generate_molecules.py \
    --num_molecules_to_generate 10000 \
    --flow_checkpoint $FLOW_LOGDIR/checkpoints-100001 \
    --random_seed $SEED \
    --save_results_json molecular_generation/results/generated_molecules_seed_${SEED}_10k.json

echo "=============================================="
echo "ALL STEPS COMPLETED for seed $SEED"
echo "End time: $(date)"
echo "=============================================="
