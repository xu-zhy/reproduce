import yaml
import pickle
import numpy as np
import torch.nn as nn

from pathlib import Path
from typing import Union, Dict
from datetime import datetime

import ray
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer

from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer
from rllib_differentiable_comms.multi_action_dist import (
    TorchHomogeneousMultiActionDistribution,
)

import vmas
from vmas import make_env

def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=vmas.Wrapper.RLLIB,
        max_steps=config["max_steps"],
        **config["scenario_config"],
    )
    return env
    
def init_ray(scenario_name: str, log_dir: Union[str, Path]):
    if not ray.is_initialized():
        ray.init(
            _temp_dir=str(Path(log_dir) / "ray"),
            local_mode=False,
        )
        print("Ray init!")
    register_env(scenario_name, lambda config: env_creator(config))
    
def get_path(scenario_name: str, algorithm_name: str, restore: bool):
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "ray_results" / scenario_name / algorithm_name

    checkpoint_dir = None
    if restore:
        latest_dir = max(
            (d for d in results_dir.iterdir() if d.is_dir()), 
            key=lambda d: datetime.strptime(d.name.split("_")[-2] + "_" + d.name.split("_")[-1], "%Y-%m-%d_%H-%M-%S")
        )

        checkpoint_dir = max(
            (c for c in latest_dir.iterdir() if c.is_dir()), 
            key=lambda c: int(c.name.split("_")[-1])
        )
        
        print(f"Restoring from {checkpoint_dir}")
    
    return project_root, checkpoint_dir

def register_model(algorithm_name: str):
    if algorithm_name == "CPPO":
        return
    # from models.ppo2 import PPO
    from models.fcnet import MyFullyConnectedNetwork
    ModelCatalog.register_custom_model(algorithm_name, MyFullyConnectedNetwork)
    ModelCatalog.register_custom_action_dist(
        "hom_multi_action", TorchHomogeneousMultiActionDistribution
    )
    
def check_args(args):
    # check algorithm name
    args.algorithm = args.algorithm.upper()
    if args.algorithm not in ["CPPO", "MAPPO", "IPPO"]:
        raise ValueError("Invalid algorithm name")
    
def load_config(algorithm_name: str, args: Dict):
    config_path = Path(__file__).parent / "models" / "config" / f"{algorithm_name.lower()}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["env"] = args.scenario_name
    config["env_config"]["scenario_name"] = args.scenario_name
    config["restore"] = args.restore
    config["callbacks"] = EvaluationCallbacks
    
    if algorithm_name == "CPPO":
        # delete model config
        del config["model"]
        
    return config
    
    
class EvaluationCallbacks(DefaultCallbacks):
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                try:
                    episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                except KeyError:
                    episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        info = episode.last_info_for()
        for a_key in info.keys():
            for b_key in info[a_key]:
                metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()
                
def get_activation_fn(name):
    # Already a callable, return as-is.
    if callable(name):
        return name

    # Infer the correct activation function from the string specifier.
    if name in ["linear", None]:
        return None
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "elu":
        return nn.ELU

    raise ValueError("Unknown activation ({}) for framework=!".format(name))

def get_checkpoint_config(checkpoint_path: Union[str, Path]):
    params_path = Path(checkpoint_path).parent.parent.parent / "params.pkl"
    with open(params_path, "rb") as f:
        config = pickle.load(f)
    return config

def get_config_trainer_and_env_from_checkpoint(algorithm, checkpoint_path, for_evaluation, config_update_fn):
    config = get_checkpoint_config(checkpoint_path)
    scenario_name = config["env"]
    init_ray(scenario_name=scenario_name)
    register_model(algorithm)

    if for_evaluation:

        # Env
        env_config = config["env_config"]
        env_config.update({"num_envs": 1})

        eval_config = config["evaluation_config"]
        eval_config.update({"callbacks": None})

        config_update = {
            "in_evaluation": True,
            "num_workers": 0,
            "num_gpus": 0,
            "num_envs_per_worker": 1,
            "callbacks": None,
            "env_config": env_config,
            "evaluation_config": eval_config
            # "explore": False,
        }
        config.update(config_update)

    if config_update_fn is not None:
        config = config_update_fn(config)

    print(f"\nConfig: {config}")

    trainer = MultiPPOTrainer(env=scenario_name, config=config)
    trainer.restore(str(checkpoint_path))
    trainer.start_config = config
    env = env_creator(config["env_config"])
    env.seed(config["seed"])

    return config, trainer, env
    
        
    
    