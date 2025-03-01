#!/bin/bash

N_ARMS=40
BUDGET=10

# N_ARMS=20
# BUDGET=5
# N_ACTIONS=10  ## for sigmoid

HORIZON=20
PREFIX=test


for SEED in {20..25}
do
    python driver.py -s $SEED -H $HORIZON -J $N_ARMS -B $BUDGET -p $PREFIX -V 50 -K 100 --rmab_type constrained
    # python driver.py -s $SEED -H $HORIZON -J $N_ARMS -B $BUDGET -p $PREFIX -V 50 -K 100 --rmab_type routing
    # python driver.py -s $SEED -H $HORIZON -J $N_ARMS -B $BUDGET -p $PREFIX -V 50 -K 100 --rmab_type scheduling
    # python driver.py -s $SEED -H $HORIZON -J $N_ARMS -B $BUDGET -p $PREFIX -V 50 -K 100 --rmab_type sigmoid -N $N_ACTIONS
done
