# multi run train.py with different algorithms

import os

for algorithm in ["IPPO"]:
    os.system(f"python train.py --algorithm {algorithm} --scenario_name transport")