#!/bin/zsh
# CHANGE ME: These are the specifications for generating data
# NUM_SEEDS=5                            # Number of random seeds to test per environment
# SEED_LOWER_BOUND=0                     # Lower bound of random seeds
# SEED_UPPER_BOUND=1000                  # Upper bound of random seeds
# SEEDS=($(shuf -i $SEED_LOWER_BOUND-$SEED_UPPER_BOUND -n $NUM_SEEDS))
# # SEEDS=("" )
# SEEDS=("133" "988" "40" "791" "967") 
# sleep 8h

SEEDS=("128" "983" "35" "786" "962")

# 0-4 5 iter  SEEDS=("133" "988" "40" "791" "967") 
#5-9 2 iter SEEDS=("128" "983" "35" "786" "962")
# 10-14 5 iter SEEDS=("128" "983" "35" "786" "962")
cnt=10
# for seed in "${SEEDS[@]}";do
#     echo "$seed" >> seeds.txt
# done

for seed in "${SEEDS[@]}";do
    python train_cadmm_multi.py --seed $seed --id $cnt
    python train_sonata_multi.py --seed $seed --id $cnt
    # python train_dsgt_multi.py --seed $seed --id $cnt
    # python train_kgt_multi.py --seed $seed --id $cnt
    

    cnt=$((cnt+1))
done
# python train_cadmm_multi.py --seed 133 --id 666
# python train_sonata_multi.py --seed 133 --id 666