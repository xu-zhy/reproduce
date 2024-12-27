'''
A script for multi-run training.
'''

import os

# for algorithm in ["CPPO", "MAPPO", "IPPO"]:
#     for n in [3, 5]:
#         os.system(f"python train.py --algorithm {algorithm} --scenario_name transport --n_agents {n}")
    # os.system(f"python train.py --algorithm {algorithm} --scenario_name transport --n_agents 4 --use_expert")
    
os.system(f"python train.py --algorithm IPPO --scenario_name transport --n_agents 4")