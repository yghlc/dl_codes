#!/usr/bin/env bash

./starter.py --epochs 50  > log_eps50.txt
#mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_withoutWeightInit.jpg

./test_batch_normalize.sh
./test_weight_init.sh
./test_dropout.sh