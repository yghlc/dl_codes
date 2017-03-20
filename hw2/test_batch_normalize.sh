#!/usr/bin/env bash

#test with dropout
./starter.py --epochs 50 --batch-Normalize > log_withBatchnormalize.txt
mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_Batchnormalize.jpg
#test without dropout
#./starter.py --epochs 50  > log_withoutWeightInit.txt
#mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_withoutBatchnormalize.jpg