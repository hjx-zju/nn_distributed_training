#!/bin/zsh
# # CHANGE ME: These are the specifications for generating data
# NUM_SEEDS=5                            # Number of random seeds to test per environment
# SEED_LOWER_BOUND=0                     # Lower bound of random seeds
# SEED_UPPER_BOUND=1000                  # Upper bound of random seeds
# SEEDS=($(shuf -i $SEED_LOWER_BOUND-$SEED_UPPER_BOUND -n $NUM_SEEDS))
# SEEDS=("" )
#sleep for 2 hours and start execute
sleep 2h
SEEDS=("133" "988" "40" "791" "967")

cnt=0
for seed in "${SEEDS[@]}";do
    echo "$seed" >> seeds.txt
done

for seed in "${SEEDS[@]}";do
    python main.py --seed $seed --id $cnt
    cnt=$((cnt+1))
done