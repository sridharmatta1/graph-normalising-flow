#!/bin/bash
#SBATCH --job-name=GNF_conditioned_confirmation
#SBATCH --output=logs/conditioned_confirmation_output.log
#SBATCH --error=logs/conditioned_confirmation_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:3

# ============================================================
# Confirmation run: enough iterations per stage to see a real,
# sustained loss trend (not just a handful of noisy steps like
# run_conditioned_smoke_test.sh), and enough generated graphs for
# a meaningful MMD readout -- but still far short of the full
# production scale (100k iterations, see run.sh). The goal here
# is to decide whether N-conditioning is worth committing to a
# full run, not to produce a final model.
# ============================================================
SEED=1
PYTHON=/home/matta/miniconda3/envs/gnf_2026/bin/python
WORKDIR=/home/matta/graph-normalising-flow
RESULTS_DIR=$WORKDIR/results/ego_small_conditioned_confirmation
DATASET=graph_rnn_ego_small
NODE_EMBEDDING_DIM=14
TRAIN_BATCH_SIZE=8

mkdir -p $RESULTS_DIR/gnn
mkdir -p $RESULTS_DIR/node_embeddings
mkdir -p $RESULTS_DIR/grevnet_conditioned
mkdir -p $RESULTS_DIR/generated_graphs
mkdir -p $WORKDIR/logs

cd $WORKDIR
export WANDB_MODE=offline

echo "=============================================="
echo "DATASET: $DATASET   SEED: $SEED   RESULTS_DIR: $RESULTS_DIR"
echo "=============================================="

# ============================================================
# STAGE 1: Train GNN autoencoder for long enough that its
# embeddings carry real structure, not near-random noise (the
# smoke test's 3 iterations weren't enough for that).
# ============================================================
echo "STAGE 1 STARTED: GNN autoencoder -- $(date)"
STAGE_START=$(date +%s)

srun $PYTHON -u $WORKDIR/run_gnn.py \
    --dataset $DATASET \
    --logdir $RESULTS_DIR/gnn \
    --attn_type dm_attn \
    --num_train_iters 2000 \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --latent_dim 2048 \
    --num_layers 3 \
    --attn_num_heads 8 \
    --num_processing_steps 10 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --log_every_n_steps 100 \
    --save_every_n_iter 500 \
    --max_checkpoints_to_keep 5 \
    --random_seed $SEED

STAGE_END=$(date +%s)
echo "STAGE 1 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

# ============================================================
# STAGE 2: Generate node embeddings from the trained autoencoder
# checkpoint -- enough examples for several distinct epochs at
# stage 3's batch size.
# ============================================================
echo "STAGE 2 STARTED: node embeddings -- $(date)"
STAGE_START=$(date +%s)

CKPT=$(ls $RESULTS_DIR/gnn/checkpoints-*.index 2>/dev/null | sort -V | tail -1 | sed 's/\.index$//')
echo "Using GNN checkpoint: $CKPT"

srun $PYTHON -u $WORKDIR/generate_grevnet_training_data.py \
    --dataset $DATASET \
    --checkpoint $CKPT \
    --output_file $RESULTS_DIR/node_embeddings/embeddings \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_examples 5000 \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --random_seed $SEED \
    --run_number $SEED

STAGE_END=$(date +%s)
echo "STAGE 2 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

# ============================================================
# STAGE 3: Train the N-conditioned GNF (NConditionedGNFBlock +
# conditional prior) on those embeddings, for long enough to see
# a real loss trend. Gradient clipping is on this time -- the
# smoke test showed loss climbing fast (720K -> 17.9B over just 4
# iterations) once NaN/Inf stopped masking it, and clipping is the
# standard first thing to try against that kind of early blow-up.
#
# Note: --sample_size is left at its default (32), NOT
# TRAIN_BATCH_SIZE -- generate_graphs.py in stage 4 hard-codes a
# batch size of 32 when restoring 'sample_n_node_placeholder' from
# the checkpoint, so this must match or stage 4 will fail with a
# shape mismatch.
# ============================================================
echo "STAGE 3 STARTED: N-conditioned GNF -- $(date)"
STAGE_START=$(date +%s)

srun $PYTHON -u $WORKDIR/train_grevnet_conditioned_with_data.py \
    --dataset $DATASET \
    --train_data_dir $RESULTS_DIR/node_embeddings \
    --logdir $RESULTS_DIR/grevnet_conditioned \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --num_coupling_layers 12 \
    --latent_dim 2048 \
    --use_batch_norm True \
    --n_embed_dim 32 \
    --prior_kl_weight 0.1 \
    --weight_sharing True \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --train_epochs 5 \
    --num_train_iters 1500 \
    --clip_gradient_by_norm True \
    --clip_gradient_norm 10 \
    --save_every_n_steps 250 \
    --log_every_n_steps 50 \
    --random_seed $SEED

STAGE_END=$(date +%s)
echo "STAGE 3 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

# ============================================================
# STAGE 4: Generate graphs from the trained conditioned GNF
# checkpoint and compute MMD against the test set (built into
# generate_graphs.py already). number_to_generate counts BATCHES
# of 32, so 8 -> 256 total graphs, enough for a meaningful MMD
# readout (not directly comparable to the paper's fully-trained
# numbers at this iteration count, just needs to be finite and
# not wildly degenerate).
# ============================================================
echo "STAGE 4 STARTED: generate graphs -- $(date)"
STAGE_START=$(date +%s)

srun $PYTHON -u $WORKDIR/generate_graphs.py \
    --dataset $DATASET \
    --ckpt_dir $RESULTS_DIR/grevnet_conditioned \
    --output_dir $RESULTS_DIR/generated_graphs \
    --node_embedding_dim $NODE_EMBEDDING_DIM \
    --number_to_generate 8

STAGE_END=$(date +%s)
echo "STAGE 4 COMPLETED -- $(date) -- $((STAGE_END - STAGE_START))s"

echo "=============================================="
echo "CONFIRMATION RUN DONE for seed $SEED"
echo "=============================================="
