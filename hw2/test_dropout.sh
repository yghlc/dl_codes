#!/bin/bash

#test with dropout
./starter.py --epochs 20 --dropout-fc > log_withDropout.txt
mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_withDropout.jpg
#test without dropout
./starter.py --epochs 20  > log_withoutDropout.txt
mv training_curve_and_accuarcy.jpg training_curve_and_accuarcy_withoutDropout.jpg


