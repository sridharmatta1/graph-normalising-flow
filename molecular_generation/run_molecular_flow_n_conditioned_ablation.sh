#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_ablation
#SBATCH --output=logs/molecular_flow_n_conditioned_ablation_output.log
#SBATCH --error=logs/molecular_flow_n_conditioned_ablation_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Ablation: is the N-conditioned flow's lower uniqueness caused by
# conditioning on N, or by the FiLM coupling networks it comes bundled
# with (one hidden layer of width 256 with mean-pooling, versus the
# unconditioned flow's self-attention GNN of width 512)?
#
# Same architecture, hyperparameters, seed, early stopping and generation
# protocol as run_molecular_flow_n_conditioned_earlystop.sh, with ONE
# change: --ablate_condition replaces the N-embedding (coupling layers and
# prior) with a constant, so the flow cannot see N. N is still sampled
# from the training histogram at generation time (it fixes how many atoms
# are drawn), exactly as for the unconditioned flow.
#
# Compare against (all single runs, seed 12345, 10,000 molecules, strict):
#   N-conditioned early-stop : uniqueness 84.0 / novelty 87.3 / N.U.V. 73.04
#       at sigma_scale 1.25  : uniqueness 87.6 / novelty 87.6 / N.U.V. 76.12
#   unconditioned (5-seed)   : uniqueness 93.36 / novelty 82.97 / N.U.V. 76.34
# If this ablated flow lands near 84%, the gap is the architecture, not
# conditioning. If it lands near 93%, conditioning is what costs diversity.
#
# Needs the val embeddings from the early-stop run
# (molecular_generation/data/val_embeddings). Writes only new files.
# ============================================================
SEED=12345
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_LOGDIR=molecular_generation/test_runs/flow_n_conditioned_ablation
VAL_DIR=molecular_generation/data/val_embeddings

mkdir -p $WORKDIR/logs

cd $WORKDIR

export WANDB_MODE=offline

if ! ls $VAL_DIR/embeddings_* >/dev/null 2>&1; then
    echo "ERROR: no validation embeddings in $VAL_DIR. Run run_molecular_flow_n_conditioned_earlystop.sh first."
    exit 1
fi

echo "=============================================="
echo "Ablation (condition removed): training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    --ablate_condition \
    --train_data_dir molecular_generation/data \
    --val_data_dir $VAL_DIR \
    --logdir $FLOW_LOGDIR \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --num_train_iters 200000 \
    --train_epochs 70 \
    --train_batch_size 32 \
    --eval_every_n_steps 2000 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.05 \
    --random_seed $SEED \
    --log_every_n_steps 100 \
    --save_every_n_steps 2000 \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_n_conditioned_qm9_ablation"

echo "=============================================="
echo "Training stopped"
echo "End time: $(date)"
echo "=============================================="

for SCALE in 1.0 1.25; do
    echo "=============================================="
    echo "Generating 10,000 molecules from the best-validation checkpoint, sigma_scale=$SCALE"
    echo "Start time: $(date)"
    echo "=============================================="

    srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
        --ablate_condition \
        --num_molecules_to_generate 10000 \
        --flow_checkpoint $FLOW_LOGDIR/best/best \
        --target_n 0 \
        --sigma_scale $SCALE \
        --random_seed $SEED \
        --save_results_json molecular_generation/results/generated_molecules_n_conditioned_ablation_sigma${SCALE}_10k.json
done

echo "=============================================="
echo "ALL STEPS COMPLETED"
echo "End time: $(date)"
echo "=============================================="
