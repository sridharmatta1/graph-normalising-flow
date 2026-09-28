#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_n_cond_5seed
#SBATCH --output=logs/molecular_flow_n_conditioned_5seed_output_%a.log
#SBATCH --error=logs/molecular_flow_n_conditioned_5seed_error_%a.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --array=1-5

# ============================================================
# 5-seed evaluation of the N-CONDITIONED molecular flow -- the
# conditioned counterpart of run_molecular_flow_5seed.sh (the
# unconditioned flow's own 5-seed evaluation). Same cost-saving
# structure: Phase 2 (frozen encoder/decoder) and Phase 3
# (extracted embeddings) don't depend on this seed and are
# reused unchanged across all 5 -- only Phase 4/5 (the
# N-conditioned flow itself) gets retrained per seed, then
# generation runs once against each seed's resulting checkpoint.
#
# Each seed:
#   1. Trains train_molecular_flow_n_conditioned.py with
#      --random_seed=$SEED, identical hyperparameters to the
#      original single-seed run
#      (run_molecular_flow_n_conditioned_full.sh) otherwise.
#   2. Generates 10,000 molecules from that seed's checkpoint,
#      with N sampled from the training distribution
#      (target_n=0, the default) -- the same "overall, mixed-N"
#      setting used for the original single-seed N-conditioned
#      comparison, so this 5-seed result is directly comparable
#      to both the unconditioned 5-seed table and MoFlow's own
#      mean +/- std over 5 runs.
#
# Aggregate afterwards with:
#   python molecular_generation/aggregate_seed_results.py \
#       --results_json_pattern "molecular_generation/results/generated_molecules_n_conditioned_seed_{}_10k.json"
# ============================================================
SEED=$SLURM_ARRAY_TASK_ID
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_LOGDIR=molecular_generation/test_runs/flow_n_conditioned_seed_$SEED

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular GNF flow (N-conditioned): seed $SEED"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    --train_data_dir molecular_generation/data \
    --logdir $FLOW_LOGDIR \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --num_train_iters 100000 \
    --train_batch_size 32 \
    --random_seed $SEED \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_n_conditioned_qm9_seed${SEED}"

echo "=============================================="
echo "Flow training complete (seed $SEED)"
echo "End time: $(date)"
echo "=============================================="

echo "=============================================="
echo "Generating 10,000 molecules (seed $SEED)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
    --num_molecules_to_generate 10000 \
    --flow_checkpoint $FLOW_LOGDIR/checkpoints-100001 \
    --random_seed $SEED \
    --save_results_json molecular_generation/results/generated_molecules_n_conditioned_seed_${SEED}_10k.json

echo "=============================================="
echo "ALL STEPS COMPLETED for seed $SEED"
echo "End time: $(date)"
echo "=============================================="
