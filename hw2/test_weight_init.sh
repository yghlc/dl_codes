#!/usr/bin/env bash

#test with dropout
./starter.py --epochs 50 --kaiming-weight-init > log_withWeightInit.txt
mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_WeightInit.jpg
#test without dropout
#./starter.py --epochs 50  > log_withoutWeightInit.txt
#mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_withoutWeightInit.jpg