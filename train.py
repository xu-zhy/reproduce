import pickle
import argparse

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer

from utils import init_ray, get_path, register_model
from utils import check_args, load_config


def train(
    scenario_name: str,
    algorithm: str,
    restore: bool=False,
):
    project_root, checkpoint_path = get_path(scenario_name, algorithm, restore)
    
    init_ray(scenario_name, project_root)
    
    if restore:
        params_path = checkpoint_path.parent / "params.pkl"
        with open(params_path, "rb") as f:
            restore_config = pickle.load(f)
    
    register_model(algorithm)
    config = load_config(algorithm, args) if not restore else restore_config

    tune.run(
        PPOTrainer if algorithm == "CPPO" else MultiPPOTrainer,
        name=algorithm,
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean", 
        local_dir=str(project_root / "ray_results" / scenario_name),
        stop={"training_iteration": 400},
        restore=str(checkpoint_path) if restore else None,
        config=config,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMAS Reproduce')
    parser.add_argument('--scenario_name', type=str, default="wheel")
    parser.add_argument('--algorithm', type=str, default="MAPPO")
    parser.add_argument('--restore', action="store_true")
    args = parser.parse_args()
    check_args(args)
    
    train(
        scenario_name=args.scenario_name,
        algorithm=args.algorithm,
        restore=args.restore,
    )