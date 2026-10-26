#!/bin/bash
# Submits one SLURM job per architecture variant (A-E) so they run side by side.
#
#   bash molecular_generation/submit_architecture_sweep.sh smoke        # all five, ~minutes each
#   bash molecular_generation/submit_architecture_sweep.sh full         # all five, hours each
#   bash molecular_generation/submit_architecture_sweep.sh full A C     # only some
#
# Run `smoke` first: it confirms each variant builds, trains, validates,
# checkpoints, restores and generates before you spend hours on it.
MODE=${1:?usage: submit_architecture_sweep.sh smoke|full [A B C D E]}
shift
ARCHS=${@:-A B C D E}

cd /home/matta/graph-normalising-flow || exit 1
mkdir -p logs

for A in $ARCHS; do
    sbatch --job-name=arch_${A}_${MODE} \
           --export=ALL,ARCH=$A,MODE=$MODE \
           molecular_generation/run_molecular_flow_arch_variant.sh
done
echo "Submitted: $ARCHS ($MODE). Logs: logs/arch_<X>_${MODE}_{output,error}.log"
