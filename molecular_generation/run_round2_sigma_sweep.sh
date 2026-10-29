#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# Generation-only sigma_scale sweep on a finished round-2 run's best
# checkpoint (run_molecular_flow_round2.sh). No training.
#
#   sbatch --job-name=sweep_NZ15 --export=ALL,CONFIG=NZ15,SCALES="1.5 1.75 2.0" \
#       molecular_generation/run_round2_sigma_sweep.sh
#
# CONFIG selects the architecture flags (must match training: the same
# mapping as run_molecular_flow_round2.sh); SEED=1..4 with CONFIG=A reads a
# plain-A seed run. Same protocol as every other table: 10,000 molecules,
# target_n=0, same seed for every scale so the comparison is paired.
# Writes only new files, named like the round-2 generation results.
# ============================================================
CONFIG=${CONFIG:?set CONFIG to A L24 NZ05 NZ10 NZ15 WIDE PERM LONG}
SCALES=${SCALES:-"1.5 1.75 2.0"}
SEED=${SEED:-12345}

ARCH_FLAGS="--coupling_arch attn_film"
NUM_LAYERS=12
case $CONFIG in
    A|NZ05|NZ10|NZ15|LONG) ;;
    L24)  NUM_LAYERS=24 ;;
    WIDE) ARCH_FLAGS="$ARCH_FLAGS --coupling_attn_kq_dim 128 --coupling_attn_v_dim 128 --coupling_attn_num_heads 4 --coupling_attn_concat_dim 128 --coupling_attn_mlp_width 1024" ;;
    PERM) ARCH_FLAGS="$ARCH_FLAGS --channel_permutation" ;;
    *) echo "CONFIG must be one of A L24 NZ05 NZ10 NZ15 WIDE PERM LONG"; exit 1 ;;
esac

TAG=$CONFIG
if [ "$SEED" != "12345" ]; then TAG=${CONFIG}_s${SEED}; fi

PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_CKPT=molecular_generation/test_runs/flow_r2_$TAG/best/best

cd $WORKDIR
mkdir -p logs molecular_generation/results

if ! ls ${FLOW_CKPT}.index >/dev/null 2>&1; then
    echo "ERROR: no checkpoint at $FLOW_CKPT (has training for $TAG finished?)"
    exit 1
fi

FAILED=0
for SCALE in $SCALES; do
    echo "=============================================="
    echo "$TAG: generating 10,000 molecules, sigma_scale=$SCALE"
    echo "Start time: $(date)"
    echo "=============================================="

    srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
        $ARCH_FLAGS --flow_num_coupling_layers $NUM_LAYERS \
        --num_molecules_to_generate 10000 \
        --flow_checkpoint $FLOW_CKPT \
        --target_n 0 \
        --sigma_scale $SCALE \
        --random_seed $SEED \
        --save_results_json molecular_generation/results/generated_molecules_r2_${TAG}_full_sigma${SCALE}_10000.json
    STATUS=$?
    echo "Generation (sigma_scale=$SCALE) exited with status $STATUS"
    if [ $STATUS -ne 0 ]; then FAILED=1; fi
done

echo "=============================================="
if [ $FAILED -ne 0 ]; then
    echo "SWEEP $TAG FAILED: a generation step exited non-zero"
    exit 1
fi
echo "SWEEP $TAG COMPLETED: $(date)"
echo "=============================================="
