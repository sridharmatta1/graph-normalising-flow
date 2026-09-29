#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_prop_cond_smoke
#SBATCH --output=logs/molecular_flow_property_conditioned_smoke_output.log
#SBATCH --error=logs/molecular_flow_property_conditioned_smoke_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Phase 4 extension smoke test: short run (2000 iters) of the
# jointly N+property-conditioned flow, before committing to a
# full 100k-iteration run -- same discipline as
# run_molecular_flow_n_conditioned_smoke.sh, since this is
# another brand-new architecture combination (NPropertyEmbedding
# feeding the same FiLM/conditional-prior machinery) that's
# never run against a real TF graph.
#
# REQUIRES property-tagged embeddings to already exist --
# generate them first with:
#   python molecular_generation/generate_molecular_embeddings.py \
#       --property_name mu \
#       --output_file molecular_generation/data/property_embeddings/embeddings
#
# Watch for in the log (same checks as the N-only smoke test):
# total_loss/log_det_jacobian should decrease/stay bounded; output
# norm should stay ~6-8; prior_sigma_mean should move away from
# ~1.0 without collapsing toward 0.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

echo "=============================================="
echo "Molecular GNF flow (N+property-conditioned): smoke test"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_property_conditioned.py \
    --train_data_dir molecular_generation/data/property_embeddings \
    --logdir molecular_generation/test_runs/flow_property_conditioned_smoke \
    --property_name mu \
    --property_mean 2.6520 \
    --property_std 1.4035 \
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
    --wandb_run_name "molecular_flow_property_conditioned_smoke"

echo "=============================================="
echo "Smoke test complete"
echo "End time: $(date)"
echo "=============================================="
