#!/bin/bash
#SBATCH --job-name=GNF_molecular_flow_prop_cond_gen
#SBATCH --output=logs/molecular_flow_property_conditioned_generate_output.log
#SBATCH --error=logs/molecular_flow_property_conditioned_generate_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Generation pass for the completed N+property-conditioned flow
# (run_molecular_flow_property_conditioned_full.sh, trained to
# 100000 iterations, checkpoints-100001).
#
# Mirrors the N-conditioned 5-seed run's generation step: 10,000
# molecules, target_n=0 (N sampled from the training distribution,
# same "overall, mixed-N" setting used for the unconditioned and
# N-conditioned tables) so this is directly comparable to those.
#
# target_property is set to the training mean (2.6520 Debye) as
# the baseline check -- confirms the model produces sane output
# when asked for a "typical" dipole moment before testing more
# extreme values.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_LOGDIR=molecular_generation/test_runs/flow_property_conditioned

mkdir -p $WORKDIR/logs

cd $WORKDIR

echo "=============================================="
echo "Generating 10,000 molecules (property-conditioned, mu=2.6520)"
echo "Start time: $(date)"
echo "=============================================="

srun $PYTHON -u molecular_generation/generate_molecules_property_conditioned.py \
    --num_molecules_to_generate 10000 \
    --flow_checkpoint $FLOW_LOGDIR/checkpoints-100001 \
    --target_n 0 \
    --target_property 2.6520 \
    --property_mean 2.6520 \
    --property_std 1.4035 \
    --random_seed 12345 \
    --save_results_json molecular_generation/results/generated_molecules_property_conditioned_mu2.65_10k.json

echo "=============================================="
echo "Generation complete"
echo "End time: $(date)"
echo "=============================================="
