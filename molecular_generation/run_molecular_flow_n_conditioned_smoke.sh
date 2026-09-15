#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_n_cond_smoke
#SBATCH --output=logs/molecular_flow_n_conditioned_smoke_output.log
#SBATCH --error=logs/molecular_flow_n_conditioned_smoke_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 4/5 smoke test: short run (2000 iters) of the new
# N-conditioned flow (NConditionedMolecularGNFBlock + conditional
# prior), before committing to the full 100k-iteration run.
#
# This is a brand-new architecture combination (FiLM-conditioned
# coupling networks + a learned N(mu(N), sigma(N)) prior, wired
# together for the first time for molecular embeddings) that has
# never run against a real TF graph -- this project's dev machine
# can't install TF1/graph_nets. train_molecular_flow.py's own
# history (plain GNFBlock -> NaN, max_log_scale=2.0 -> still NaN,
# 0.25 -> stable) is exactly why this checks for the same failure
# signature (loss/log_det_jacobian exploding, output norm blowing
# up) on a short run before spending a full 100k-iteration budget
# on it.
#
# Watch for in the log: total_loss/log_det_jacobian should
# decrease/stay bounded, not grow every single step; grevnet
# output norm should stay in a sane range (~5-10, matching the
# real Phase 3 embeddings' own scale), not run into the thousands+;
# prior_sigma_mean should move away from its ~1.0 init but not
# collapse toward 0 (which would blow up conditional_prior_log_prob).
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular GNF flow (N-conditioned): smoke test"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    --train_data_dir molecular_generation/data \
    --logdir molecular_generation/test_runs/flow_n_conditioned_smoke \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --num_train_iters 2000 \
    --train_batch_size 32 \
    --log_every_n_steps 50 \
    --save_every_n_steps 1000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_n_conditioned_smoke"

echo "=============================================="
echo "Smoke test complete"
echo "End time: $(date)"
echo "=============================================="
