#!/bin/bash
#SBATCH --job-name=GNF_real_vs_generated_emb
#SBATCH --output=logs/real_vs_generated_embeddings_output.log
#SBATCH --error=logs/real_vs_generated_embeddings_error.log
#SBATCH --mail-user=matta@uni-hildesheim.de
#SBATCH --mail-type=ALL
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1

# ============================================================
# End-to-end check, before the decoder: do embeddings sampled from each
# flow match the REAL encoder embeddings (spread, correlations,
# between-molecule variation, MMD), and is the inverse flow exact?
#
#   1. N-conditioned, early-stopped best, sigma_scale 1.0 (as generated)
#   2. N-conditioned, early-stopped best, sigma_scale 1.5 (the widened
#      sampling that raised uniqueness)
#   3. N-conditioned, original 100k, sigma_scale 1.0
#   4. Unconditioned, 100k, temperature 1.0 (reference)
#
# Reads existing checkpoints and training embeddings; writes only new
# JSON files.
# ============================================================
PYTHON=/home/matta/miniconda3/envs/gnf_molecular/bin/python
WORKDIR=/home/matta/graph-normalising-flow
RUNS=molecular_generation/test_runs
OUT=molecular_generation/results
SCRIPT=molecular_generation/compare_real_vs_generated_embeddings_molecular.py

mkdir -p $WORKDIR/logs $WORKDIR/$OUT

cd $WORKDIR

echo "##### 1/4 N-conditioned early-stop best, sigma_scale 1.0 #####"
srun $PYTHON -u $SCRIPT --model_kind n_conditioned \
    --flow_checkpoint $RUNS/flow_n_conditioned_earlystop/best/best \
    --sigma_scale 1.0 \
    --save_summary_json $OUT/real_vs_generated_n_cond_earlystop_s1.0.json

echo "##### 2/4 N-conditioned early-stop best, sigma_scale 1.5 #####"
srun $PYTHON -u $SCRIPT --model_kind n_conditioned \
    --flow_checkpoint $RUNS/flow_n_conditioned_earlystop/best/best \
    --sigma_scale 1.5 \
    --save_summary_json $OUT/real_vs_generated_n_cond_earlystop_s1.5.json

echo "##### 3/4 N-conditioned original 100k, sigma_scale 1.0 #####"
srun $PYTHON -u $SCRIPT --model_kind n_conditioned \
    --flow_checkpoint $RUNS/flow_n_conditioned/checkpoints-100001 \
    --sigma_scale 1.0 \
    --save_summary_json $OUT/real_vs_generated_n_cond_100k_s1.0.json

echo "##### 4/4 Unconditioned 100k, temperature 1.0 #####"
srun $PYTHON -u $SCRIPT --model_kind unconditioned \
    --flow_checkpoint $RUNS/flow_full/checkpoints-100001 \
    --sample_temperature 1.0 \
    --save_summary_json $OUT/real_vs_generated_uncond_100k_t1.0.json

echo "ALL DONE: $(date)"
