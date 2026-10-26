#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# One architecture variant of the N-conditioned molecular flow, selected by
# the ARCH environment variable. Launch through
# submit_architecture_sweep.sh (it sets --job-name, which names the logs).
#
#   ARCH=A  attn_film          unconditioned flow's self-attention GNN + FiLM
#   ARCH=B  deep_film          width 512, 3 residual message-passing rounds
#   ARCH=C  interaction_film   pairwise interaction network, 2 rounds
#   ARCH=D  film + channel_permutation   original block, mixed channels
#   ARCH=E  spline_film        original FiLM net predicting spline parameters
#
#   MODE=smoke  400 iterations + a quick validation + 64 generated molecules:
#               checks the variant builds, trains, validates, saves, restores
#               and generates, and reports the time per step.
#   MODE=full   same protocol as run_molecular_flow_n_conditioned_earlystop.sh
#               (seed 12345, validation every 2000 steps, patience 10,
#               min_delta 0.05, hard cap 200k), then 10,000 molecules at
#               sigma_scale 1.0 and 1.25.
#
# Everything except the architecture flags is identical across variants and
# to the baseline early-stop run, so differences are due to the architecture.
# Writes only to new directories/files.
# ============================================================
ARCH=${ARCH:?set ARCH to one of A B C D E}
MODE=${MODE:?set MODE to smoke or full}

case $ARCH in
    A) ARCH_FLAGS="--coupling_arch attn_film" ;;
    B) ARCH_FLAGS="--coupling_arch deep_film" ;;
    C) ARCH_FLAGS="--coupling_arch interaction_film" ;;
    D) ARCH_FLAGS="--coupling_arch film --channel_permutation" ;;
    E) ARCH_FLAGS="--coupling_arch spline_film" ;;
    *) echo "ARCH must be one of A B C D E"; exit 1 ;;
esac

SEED=12345
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
VAL_DIR=molecular_generation/data/val_embeddings

cd $WORKDIR
mkdir -p logs molecular_generation/results
export WANDB_MODE=offline

if ! ls $VAL_DIR/embeddings_* >/dev/null 2>&1; then
    echo "ERROR: no validation embeddings in $VAL_DIR. Run run_molecular_flow_n_conditioned_earlystop.sh first."
    exit 1
fi

if [ "$MODE" = "smoke" ]; then
    FLOW_LOGDIR=molecular_generation/test_runs/arch_smoke_$ARCH
    TRAIN_FLAGS="--num_train_iters 400 --train_epochs 2 --eval_every_n_steps 200 --val_max_batches 40 --early_stopping_patience 0 --save_every_n_steps 200 --log_every_n_steps 50"
    NUM_GEN=64
    SCALES="1.0"
elif [ "$MODE" = "full" ]; then
    FLOW_LOGDIR=molecular_generation/test_runs/flow_arch_$ARCH
    TRAIN_FLAGS="--num_train_iters 200000 --train_epochs 70 --eval_every_n_steps 2000 --early_stopping_patience 10 --early_stopping_min_delta 0.05 --save_every_n_steps 2000 --log_every_n_steps 100"
    NUM_GEN=10000
    SCALES="1.0 1.25"
else
    echo "MODE must be smoke or full"; exit 1
fi

echo "=============================================="
echo "Architecture $ARCH ($ARCH_FLAGS), mode $MODE: training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    $ARCH_FLAGS \
    --train_data_dir molecular_generation/data \
    --val_data_dir $VAL_DIR \
    --logdir $FLOW_LOGDIR \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --train_batch_size 32 \
    --random_seed $SEED \
    $TRAIN_FLAGS \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_arch_${ARCH}_${MODE}"
STATUS=$?

echo "=============================================="
echo "Training exited with status $STATUS"
echo "End time: $(date)"
echo "=============================================="
if [ $STATUS -ne 0 ]; then
    exit $STATUS
fi

for SCALE in $SCALES; do
    echo "=============================================="
    echo "Generating $NUM_GEN molecules from the best-validation checkpoint, sigma_scale=$SCALE"
    echo "Start time: $(date)"
    echo "=============================================="

    srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
        $ARCH_FLAGS \
        --num_molecules_to_generate $NUM_GEN \
        --flow_checkpoint $FLOW_LOGDIR/best/best \
        --target_n 0 \
        --sigma_scale $SCALE \
        --random_seed $SEED \
        --save_results_json molecular_generation/results/generated_molecules_arch_${ARCH}_${MODE}_sigma${SCALE}_${NUM_GEN}.json
done

echo "=============================================="
echo "ARCH $ARCH ($MODE) COMPLETED"
echo "End time: $(date)"
echo "=============================================="
