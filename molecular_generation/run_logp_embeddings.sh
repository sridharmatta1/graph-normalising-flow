#!/bin/bash
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_output.log
#SBATCH --error=logs/%x_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL

# ============================================================
# Step 1 of the logP-conditioned flow: encode the QM9 train and val splits with
# the frozen Phase 2 encoder and tag every molecule with its Crippen logP
# (derived_properties.py, computed from the molecular graph). Writes
# property-tagged 3-tuple chunks (node_embeddings, n_node, logp) to two NEW
# directories; the existing embeddings are not touched.
#
#   sbatch --job-name=logp_embeddings molecular_generation/run_logp_embeddings.sh
#
# Takes a few minutes. Afterwards: run_molecular_flow_logp.sh.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
TRAIN_OUT=molecular_generation/data/logp_embeddings
VAL_OUT=molecular_generation/data/logp_val_embeddings

cd $WORKDIR
mkdir -p logs $TRAIN_OUT $VAL_OUT

echo "##### train split, property logp #####"
srun $PYTHON -u molecular_generation/generate_molecular_embeddings.py \
    --property_name logp \
    --output_file $TRAIN_OUT/embeddings
TRAIN_STATUS=$?

echo "##### val split, property logp #####"
srun $PYTHON -u molecular_generation/generate_molecular_embeddings.py \
    --split val \
    --property_name logp \
    --output_file $VAL_OUT/embeddings
VAL_STATUS=$?

echo "train exit $TRAIN_STATUS, val exit $VAL_STATUS"
ls $TRAIN_OUT $VAL_OUT
if [ $TRAIN_STATUS -ne 0 ] || [ $VAL_STATUS -ne 0 ]; then
    echo "LOGP EMBEDDINGS FAILED"
    exit 1
fi
echo "LOGP EMBEDDINGS COMPLETED"
