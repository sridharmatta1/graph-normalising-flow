#!/bin/bash
# Submits one SLURM job per round-2 configuration so they run side by side.
#
#   bash molecular_generation/submit_round2.sh smoke               # the six configs, ~minutes each
#   bash molecular_generation/submit_round2.sh full                # the six configs, hours each
#   bash molecular_generation/submit_round2.sh full seeds          # plain A, seeds 1-4
#   bash molecular_generation/submit_round2.sh full L24 NZ05       # only some
#   bash molecular_generation/submit_round2.sh full NZ05:seeds     # NZ05, seeds 1-4
#
# Configs: L24 NZ05 NZ10 NZ15 WIDE PERM LONG (see run_molecular_flow_round2.sh).
# Run `smoke` first: it confirms every config builds, trains, validates,
# restores, generates and diagnoses before you spend hours on it.
MODE=${1:?usage: submit_round2.sh smoke|full [L24 NZ05 NZ10 NZ15 WIDE PERM LONG seeds <config>:seeds]}
shift
CONFIGS=${@:-L24 NZ05 NZ15 WIDE PERM LONG}

cd /home/matta/graph-normalising-flow || exit 1
mkdir -p logs

for C in $CONFIGS; do
    if [ "$C" = "seeds" ] || [ "${C#*:}" = "seeds" ]; then
        BASE=A
        if [ "$C" != "seeds" ]; then BASE=${C%%:*}; fi
        for S in 1 2 3 4; do
            sbatch --job-name=r2_${BASE}_s${S}_${MODE} \
                   --export=ALL,CONFIG=$BASE,SEED=$S,MODE=$MODE \
                   molecular_generation/run_molecular_flow_round2.sh
        done
    else
        sbatch --job-name=r2_${C}_${MODE} \
               --export=ALL,CONFIG=$C,MODE=$MODE \
               molecular_generation/run_molecular_flow_round2.sh
    fi
done
echo "Submitted: $CONFIGS ($MODE). Logs: logs/r2_<config>_${MODE}_{output,error}.log"
