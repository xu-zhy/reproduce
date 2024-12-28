'''
A script for multi-run training.
'''

import os

for algorithm in ["MAPPO"]:
    os.system(f"python train.py --algorithm {algorithm} --scenario_name transport")