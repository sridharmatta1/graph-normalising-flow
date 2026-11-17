#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# Generation only (no training): asks a finished logP-conditioned flow
# (run_molecular_flow_logp.sh) for a whole RANGE of logP values and scores
# every strictly valid generated molecule with RDKit, so the achieved logP can
# be compared with the requested one across the full logP range, including
# values far outside the five percentile targets of the main run.
#
#   sbatch --job-name=logp_range --export=ALL \
#       molecular_generation/run_logp_range.sh
#
# Optional environment: SEED (default 12345; reads that seed's checkpoint),
# TARGETS (comma-separated requested logP values), NUM_PER_TARGET (default
# 500), SCALE (sigma scale, default 1.0). Writes one new JSON and prints a
# requested-vs-generated table.
# ============================================================
SEED=${SEED:-12345}
TARGETS=${TARGETS:-"-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3"}
NUM_PER_TARGET=${NUM_PER_TARGET:-500}
SCALE=${SCALE:-1.0}

# Training-set logP (all 105,632 molecules), as in run_molecular_flow_logp.sh.
PROP_MEAN=0.3345
PROP_STD=0.9459

TAG=logp_A
if [ "$SEED" != "12345" ]; then TAG=${TAG}_s${SEED}; fi

PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_CKPT=molecular_generation/test_runs/flow_$TAG/best/best
OUT=molecular_generation/results/generated_${TAG}_range_sigma${SCALE}.json

cd $WORKDIR
mkdir -p logs molecular_generation/results

if ! ls ${FLOW_CKPT}.index >/dev/null 2>&1; then
    echo "ERROR: no checkpoint at $FLOW_CKPT (has training for $TAG finished?)"
    exit 1
fi

echo "=============================================="
echo "$TAG: $NUM_PER_TARGET molecules per target for logP targets $TARGETS, sigma_scale=$SCALE"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/generate_molecules_property_conditioned.py \
    --coupling_arch attn_film --flow_num_coupling_layers 12 \
    --flow_checkpoint $FLOW_CKPT \
    --property_name logp \
    --property_mean $PROP_MEAN \
    --property_std $PROP_STD \
    --target_properties="$TARGETS" \
    --num_molecules_to_generate $NUM_PER_TARGET \
    --sigma_scale $SCALE \
    --random_seed $SEED \
    --save_results_json $OUT
STATUS=$?
echo "Generation exited with status $STATUS"
if [ $STATUS -ne 0 ]; then
    echo "LOGP RANGE $TAG FAILED"
    exit 1
fi

echo "=============================================="
echo "Requested vs generated logP ($TAG)"
echo "=============================================="
$PYTHON - "$OUT" <<'EOF'
import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
by = {}
for m in d['molecules']:
    if m['connected'] and m['achieved'] is not None:
        by.setdefault(m['target'], []).append(m['achieved'])
print("requested | n valid | generated mean | std | generated - requested | within 0.25 | within 0.5")
for t in sorted(by):
    v = np.array(by[t])
    print("%9.3f | %7d | %14.3f | %.3f | %+21.3f | %10.1f%% | %9.1f%%" % (
        t, len(v), v.mean(), v.std(), v.mean() - t,
        100 * np.mean(np.abs(v - t) <= .25), 100 * np.mean(np.abs(v - t) <= .5)))
ts = np.array(sorted(by))
ms = np.array([np.mean(by[t]) for t in ts])
print("slope of generated mean vs requested over all targets: %.3f (1.0 = exact)" % np.polyfit(ts, ms, 1)[0])
EOF
echo "LOGP RANGE $TAG COMPLETED: $(date)"
