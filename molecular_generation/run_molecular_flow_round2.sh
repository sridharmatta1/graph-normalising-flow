#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# Round 2: variations on architecture A (attn_film), the best coupling
# network of round 1. Launch through submit_round2.sh (it sets --job-name,
# which names the logs). One run per CONFIG, selected by environment:
#
#   CONFIG=A      plain A with a different SEED (noise reference; seeds 1-4
#                 join the existing seed-12345 run for a 5-seed estimate)
#   CONFIG=L24    24 coupling layers instead of 12
#   CONFIG=NZ05   embedding noise 0.05 x per-dimension data std (training only)
#   CONFIG=NZ15   embedding noise 0.15 x per-dimension data std
#   CONFIG=WIDE   attention 4 heads, kq/v/concat 128, MLP width 1024
#   CONFIG=PERM   channel permutation between coupling layers
#   CONFIG=LONG   cap 300k steps, polynomial LR decay to lr/100, patience 20
#
#   MODE=smoke  400 iterations + a quick validation/probe + 64 molecules +
#               a calibration pass: checks the config builds, trains,
#               validates, restores, generates and diagnoses.
#   MODE=full   same protocol as round 1 (validation every 2000 steps,
#               patience 10, min_delta 0.05, cap 200k, 10,000 molecules at
#               sigma_scale 1.0 and 1.25) except where CONFIG says
#               otherwise, followed by the latent calibration report.
#
# Everything not named in CONFIG is identical to round 1's variant A, so a
# difference is due to that one change. The probe adds a GAP line to the
# training log (val minus train loss and z-score RMS, noise-free) at every
# validation step. Writes only to new directories/files.
# ============================================================
CONFIG=${CONFIG:?set CONFIG to A L24 NZ05 NZ15 WIDE PERM LONG}
MODE=${MODE:?set MODE to smoke or full}
SEED=${SEED:-12345}

ARCH_FLAGS="--coupling_arch attn_film"   # shared by trainer, generator, calibration
TRAIN_EXTRA=""                           # trainer only
NUM_LAYERS=12
CAP=200000
EPOCHS=70
PATIENCE=10

case $CONFIG in
    A)    ;;
    L24)  NUM_LAYERS=24 ;;
    NZ05) TRAIN_EXTRA="--embedding_noise 0.05" ;;
    NZ15) TRAIN_EXTRA="--embedding_noise 0.15" ;;
    WIDE) ARCH_FLAGS="$ARCH_FLAGS --coupling_attn_kq_dim 128 --coupling_attn_v_dim 128 --coupling_attn_num_heads 4 --coupling_attn_concat_dim 128 --coupling_attn_mlp_width 1024" ;;
    PERM) ARCH_FLAGS="$ARCH_FLAGS --channel_permutation" ;;
    LONG) TRAIN_EXTRA="--lr_type polynomial_decay"
          CAP=300000; EPOCHS=150; PATIENCE=20 ;;
    *) echo "CONFIG must be one of A L24 NZ05 NZ15 WIDE PERM LONG"; exit 1 ;;
esac

TAG=$CONFIG
if [ "$SEED" != "12345" ]; then TAG=${CONFIG}_s${SEED}; fi

PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
VAL_DIR=molecular_generation/data/val_embeddings

cd $WORKDIR
mkdir -p logs molecular_generation/results
export WANDB_MODE=offline

if ! ls $VAL_DIR/embeddings_* >/dev/null 2>&1; then
    echo "ERROR: no validation embeddings in $VAL_DIR."
    exit 1
fi

if [ "$MODE" = "smoke" ]; then
    FLOW_LOGDIR=molecular_generation/test_runs/r2_smoke_$TAG
    TRAIN_FLAGS="--num_train_iters 400 --train_epochs 2 --eval_every_n_steps 200 --val_max_batches 40 --early_stopping_patience 0 --save_every_n_steps 200 --log_every_n_steps 50 --train_probe_batches 10"
    NUM_GEN=64
    SCALES="1.0"
    CALIB_EXTRA="--max_molecules_per_n 200"
elif [ "$MODE" = "full" ]; then
    FLOW_LOGDIR=molecular_generation/test_runs/flow_r2_$TAG
    TRAIN_FLAGS="--num_train_iters $CAP --train_epochs $EPOCHS --eval_every_n_steps 2000 --early_stopping_patience $PATIENCE --early_stopping_min_delta 0.05 --save_every_n_steps 2000 --log_every_n_steps 100 --train_probe_batches 50"
    NUM_GEN=10000
    SCALES="1.0 1.25"
    CALIB_EXTRA=""
else
    echo "MODE must be smoke or full"; exit 1
fi

echo "=============================================="
echo "Round 2 $TAG ($ARCH_FLAGS $TRAIN_EXTRA), seed $SEED, mode $MODE: training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_n_conditioned.py \
    $ARCH_FLAGS $TRAIN_EXTRA \
    --train_data_dir molecular_generation/data \
    --val_data_dir $VAL_DIR \
    --logdir $FLOW_LOGDIR \
    --node_embedding_dim 64 \
    --num_coupling_layers $NUM_LAYERS \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --train_batch_size 32 \
    --random_seed $SEED \
    $TRAIN_FLAGS \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_r2_${TAG}_${MODE}"
STATUS=$?

echo "=============================================="
echo "Training exited with status $STATUS"
echo "End time: $(date)"
echo "=============================================="
if [ $STATUS -ne 0 ]; then
    exit $STATUS
fi

FAILED=0
for SCALE in $SCALES; do
    echo "=============================================="
    echo "Generating $NUM_GEN molecules from the best-validation checkpoint, sigma_scale=$SCALE"
    echo "Start time: $(date)"
    echo "=============================================="

    srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
        $ARCH_FLAGS --flow_num_coupling_layers $NUM_LAYERS \
        --num_molecules_to_generate $NUM_GEN \
        --flow_checkpoint $FLOW_LOGDIR/best/best \
        --target_n 0 \
        --sigma_scale $SCALE \
        --random_seed $SEED \
        --save_results_json molecular_generation/results/generated_molecules_r2_${TAG}_${MODE}_sigma${SCALE}_${NUM_GEN}.json
    GEN_STATUS=$?
    echo "Generation (sigma_scale=$SCALE) exited with status $GEN_STATUS"
    if [ $GEN_STATUS -ne 0 ]; then FAILED=1; fi
done

echo "=============================================="
echo "Latent calibration of the best checkpoint"
echo "=============================================="
srun $PYTHON -u molecular_generation/inspect_latent_calibration.py \
    --model_kind n_conditioned \
    $ARCH_FLAGS --flow_num_coupling_layers $NUM_LAYERS $CALIB_EXTRA \
    --flow_checkpoint $FLOW_LOGDIR/best/best \
    --save_summary_json molecular_generation/results/latent_calibration_r2_${TAG}_${MODE}.json
CAL_STATUS=$?
echo "Calibration exited with status $CAL_STATUS"
if [ $CAL_STATUS -ne 0 ]; then FAILED=1; fi

echo "=============================================="
if [ $FAILED -ne 0 ]; then
    echo "ROUND2 $TAG ($MODE) FAILED: a generation or calibration step exited non-zero"
    echo "End time: $(date)"
    exit 1
fi
echo "ROUND2 $TAG ($MODE) COMPLETED"
echo "End time: $(date)"
echo "=============================================="
