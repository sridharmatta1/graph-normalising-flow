#!/bin/bash
#SBATCH --job-name=GNF_molecular_n_cond_sigma_sweep
#SBATCH --output=logs/molecular_flow_n_conditioned_sigma_sweep_output.log
#SBATCH --error=logs/molecular_flow_n_conditioned_sigma_sweep_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# Sigma-scale sweep on the early-stopped N-conditioned flow
# (run_molecular_flow_n_conditioned_earlystop.sh's best-validation
# checkpoint). No retraining: only generation changes.
#
# Question: does widening the conditional prior's sampling spread
# (sigma(N) * scale) recover uniqueness over the whole size mix,
# not just at N=5 where the 1.5 fix was first found? The baseline
# (scale 1.0) is already in
# results/generated_molecules_n_conditioned_earlystop_10k.json.
#
# Same protocol as every other table: 10,000 molecules, target_n=0
# (N sampled from the training distribution), same seed for every
# scale so the comparison is paired. Edit SCALES to sweep other
# values. Writes only new files.
# ============================================================
SCALES="1.25 1.5"
SEED=12345
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
FLOW_CKPT=molecular_generation/test_runs/flow_n_conditioned_earlystop/best/best

mkdir -p $WORKDIR/logs

cd $WORKDIR

for SCALE in $SCALES; do
    echo "=============================================="
    echo "Generating 10,000 molecules, sigma_scale=$SCALE"
    echo "Start time: $(date)"
    echo "=============================================="

    srun $PYTHON -u molecular_generation/generate_molecules_n_conditioned.py \
        --num_molecules_to_generate 10000 \
        --flow_checkpoint $FLOW_CKPT \
        --target_n 0 \
        --sigma_scale $SCALE \
        --random_seed $SEED \
        --save_results_json molecular_generation/results/generated_molecules_n_conditioned_earlystop_sigma${SCALE}_10k.json
done

echo "=============================================="
echo "ALL SCALES COMPLETED"
echo "End time: $(date)"
echo "=============================================="
