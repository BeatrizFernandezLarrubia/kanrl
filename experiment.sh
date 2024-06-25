#!/bin/sh
echo "Starting experiments with MLP..."
# Change the width of the MLP or the seed range, but make sure it is the same in both commands
python kaqn.py --multirun seed="range(5)" method=MLP width=32
echo "Starting experiments with KAN..."
python kaqn.py --multirun seed="range(5)"