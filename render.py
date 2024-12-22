import os
import sys

import cv2
import ray
import torch
import pickle
import numpy as np

from pathlib import Path
from typing import Union, Dict, Callable

from vmas import make_env
from vmas.simulator.environment import Wrapper

from ray.rllib import VectorEnv
from ray.tune import register_env
from ray.rllib.agents.ppo import PPOTrainer
from rllib_differentiable_comms.multi_trainer import MultiPPOTrainer

from utils import init_ray, register_model

def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # Scenario specific variables
        **config["scenario_config"],
    )
    return env

def init_ray(scenario_name: str, local_mode: bool = False):
    if not ray.is_initialized():
        ray.init()
        print("Ray init!")
    register_env(scenario_name, lambda config: env_creator(config))

def get_checkpoint_config(checkpoint_path: Union[str, Path]):
    params_path = Path(checkpoint_path).parent.parent.parent / "params.pkl"
    with open(params_path, "rb") as f:
        config = pickle.load(f)
    return config
    
def get_config_trainer_and_env_from_checkpoint(
    checkpoint_path: Union[str, Path],
    for_evaluation: bool = True,
    config_update_fn: Callable[[Dict], Dict] = None,
):
    config = get_checkpoint_config(checkpoint_path)
    scenario_name = config["env"]
    init_ray(scenario_name=scenario_name)
    register_model("MAPPO")

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


def rollout_episodes(
    n_episodes: int,
    render: bool,
    get_obs: bool,
    get_actions: bool,
    trainer: PPOTrainer,
    env: VectorEnv,
    action_callback=None,
):
    assert (trainer is None) != (action_callback is None)

    best_gif = None
    rewards = []
    observations = []
    actions = []

    best_reward = max(rewards, default=float("-inf"))

    for j in range(len(rewards), n_episodes):
        env.seed(j)
        frame_list = []
        observations_this_episode = []
        actions_this_episode = []
        reward_sum = 0
        observation = env.vector_reset()[0]
        i = 0
        done = False
        if render:
            frame_list.append(
                env.try_render_at(mode="rgb_array", visualize_when_rgb=True)
            )
        while not done:
            i += 1

            if get_obs:
                observations_this_episode.append(observation)

            if trainer is not None:
                action = trainer.compute_single_action(observation)
            else:
                action = action_callback(observation)

            if get_actions:
                actions_this_episode.append(action)
            obss, rews, ds, infos = env.vector_step([action])
            observation = obss[0]
            reward = rews[0]
            done = ds[0]
            info = infos[0]
            reward_sum += reward
            if render:
                frame_list.append(
                    env.try_render_at(mode="rgb_array", visualize_when_rgb=True)
                )
        print(f"Episode: {j + 1}, total reward: {reward_sum}")
        rewards.append(reward_sum)
        if reward_sum > best_reward and render:
            best_reward = reward_sum
            best_gif = frame_list.copy()
        if get_obs:
            observations.append(observations_this_episode)
        if get_actions:
            actions.append(actions_this_episode)
    print(
        f"Max reward: {np.max(rewards)}\nReward mean: {np.mean(rewards)}\nMin reward: {np.min(rewards)}"
    )

    assert len(rewards) == n_episodes
    if get_obs:
        assert len(observations) == n_episodes
    if get_actions:
        assert len(actions) == n_episodes
    if render:
        assert best_gif

    return (
        rewards,
        best_gif,
        observations,
        actions,
    )
    
def export(
    checkpoint_path: Union[str, Path],
):

    config, trainer, env = get_config_trainer_and_env_from_checkpoint(
        checkpoint_path
    )

    model_path = (
        Path(checkpoint_path).parent
        / "test.pt"
    )
    print(f"Saving model to {model_path}")

    model = trainer.get_policy().model

    torch.save(model, model_path)

def render(
    checkpoint_path: Union[str, Path],
    n_episodes: int,
):
    config, trainer, env = get_config_trainer_and_env_from_checkpoint(
        Path(checkpoint_path)
    )

    rewards, best_gif, _, _ = rollout_episodes(
        n_episodes=n_episodes,
        render=True,
        get_obs=False,
        get_actions=False,
        trainer=trainer,
        env=env,
    )
    
    name = "Test"
    video = cv2.VideoWriter(
        str(f"{name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (best_gif[0].shape[1], best_gif[0].shape[0]),
    )
    for img in best_gif:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)
    video.release()


if __name__ == "__main__":
    checkpoint_path = "C:\\Coding\\Python\\vmas_test\\reproduce\\ray_results\\transport\\MAPPO\\MultiPPOTrainer_transport_ebbd6_00000_0_2024-12-22_00-16-23\\checkpoint_000395\\policies\\default_policy"
    # export(checkpoint_path)
    render(checkpoint_path, n_episodes=1)
