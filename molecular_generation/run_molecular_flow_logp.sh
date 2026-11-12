#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# Step 2 of the logP-conditioned flow: architecture A (attn_film, 12 layers)
# conditioned jointly on N and the Crippen logP of the molecule, with the
# validation / early-stopping / gap-probe protocol of the N-conditioned study.
# Needs run_logp_embeddings.sh to have finished.
#
#   sbatch --job-name=logp_A_smoke --export=ALL,MODE=smoke \
#       molecular_generation/run_molecular_flow_logp.sh
#   sbatch --job-name=logp_A_full  --export=ALL,MODE=full \
#       molecular_generation/run_molecular_flow_logp.sh
#
# Optional environment: NOISE (training-only embedding noise, default 0 =
# plain A, e.g. NOISE=0.15), SEED (default 12345; also names the run).
# MODE=smoke: 400 iterations, a few molecules per target, checks every stage.
# MODE=full: 200k cap with early stopping, 2000 molecules per target.
#
# Generation asks for five logP targets (the 5/25/50/75/95th percentiles of the
# training set) at sigma scale 1.0 and 1.25; each generated, strictly valid
# molecule is scored with RDKit, so the report says how closely the achieved
# logP follows the request (achieved mean per target, bias, MAE, and the slope of
# achieved mean vs target: 1.0 = follows the request, 0 = ignores it).
# Writes only to new directories/files.
# ============================================================
MODE=${MODE:?set MODE to smoke or full}
NOISE=${NOISE:-0}
SEED=${SEED:-12345}

# Training-set logP (all 105,632 molecules, derived_properties.py).
PROP_MEAN=0.3345
PROP_STD=0.9459
TARGETS="-1.156,-0.319,0.299,0.955,1.948"

ARCH_FLAGS="--coupling_arch attn_film"   # shared by trainer and generator
TRAIN_EXTRA=""
TAG=logp_A
if [ "$NOISE" != "0" ]; then
    TRAIN_EXTRA="--embedding_noise $NOISE"
    TAG=logp_NZ${NOISE}
fi
if [ "$SEED" != "12345" ]; then TAG=${TAG}_s${SEED}; fi

PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
TRAIN_DIR=molecular_generation/data/logp_embeddings
VAL_DIR=molecular_generation/data/logp_val_embeddings

cd $WORKDIR
mkdir -p logs molecular_generation/results
export WANDB_MODE=offline

for D in $TRAIN_DIR $VAL_DIR; do
    if ! ls $D/embeddings_* >/dev/null 2>&1; then
        echo "ERROR: no embeddings in $D. Run run_logp_embeddings.sh first."
        exit 1
    fi
done

if [ "$MODE" = "smoke" ]; then
    FLOW_LOGDIR=molecular_generation/test_runs/logp_smoke_$TAG
    TRAIN_FLAGS="--num_train_iters 400 --train_epochs 2 --eval_every_n_steps 200 --val_max_batches 40 --early_stopping_patience 0 --save_every_n_steps 200 --log_every_n_steps 50 --train_probe_batches 10"
    NUM_PER_TARGET=32
    SCALES="1.0"
elif [ "$MODE" = "full" ]; then
    FLOW_LOGDIR=molecular_generation/test_runs/flow_$TAG
    TRAIN_FLAGS="--num_train_iters 200000 --train_epochs 70 --eval_every_n_steps 2000 --early_stopping_patience 10 --early_stopping_min_delta 0.05 --save_every_n_steps 2000 --log_every_n_steps 100 --train_probe_batches 50"
    NUM_PER_TARGET=2000
    SCALES="1.0 1.25"
else
    echo "MODE must be smoke or full"; exit 1
fi

echo "=============================================="
echo "logP-conditioned flow $TAG ($ARCH_FLAGS $TRAIN_EXTRA), seed $SEED, mode $MODE: training"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/train_molecular_flow_property_conditioned.py \
    $ARCH_FLAGS $TRAIN_EXTRA \
    --train_data_dir $TRAIN_DIR \
    --val_data_dir $VAL_DIR \
    --logdir $FLOW_LOGDIR \
    --property_name logp \
    --property_mean $PROP_MEAN \
    --property_std $PROP_STD \
    --node_embedding_dim 64 \
    --num_coupling_layers 12 \
    --hidden_dim 256 \
    --n_embed_dim 32 \
    --max_log_scale 0.25 \
    --train_batch_size 32 \
    --random_seed $SEED \
    $TRAIN_FLAGS \
    --wandb_project graph-normalising-flow \
    --wandb_run_name "molecular_flow_${TAG}_${MODE}"
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
    echo "Generating $NUM_PER_TARGET molecules per logP target from the best-validation checkpoint, sigma_scale=$SCALE"
    echo "Start time: $(date)"
    echo "=============================================="

    srun $PYTHON -u molecular_generation/generate_molecules_property_conditioned.py \
        $ARCH_FLAGS --flow_num_coupling_layers 12 \
        --flow_checkpoint $FLOW_LOGDIR/best/best \
        --property_name logp \
        --property_mean $PROP_MEAN \
        --property_std $PROP_STD \
        --target_properties="$TARGETS" \
        --num_molecules_to_generate $NUM_PER_TARGET \
        --sigma_scale $SCALE \
        --random_seed $SEED \
        --save_results_json molecular_generation/results/generated_${TAG}_${MODE}_sigma${SCALE}.json
    GEN_STATUS=$?
    echo "Generation (sigma_scale=$SCALE) exited with status $GEN_STATUS"
    if [ $GEN_STATUS -ne 0 ]; then FAILED=1; fi
done

echo "=============================================="
if [ $FAILED -ne 0 ]; then
    echo "LOGP $TAG ($MODE) FAILED: a generation step exited non-zero"
    echo "End time: $(date)"
    exit 1
fi
echo "LOGP $TAG ($MODE) COMPLETED"
echo "End time: $(date)"
echo "=============================================="
