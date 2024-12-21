import time
from typing import Type

import torch

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video


def run_heuristic(
    scenario_name: str,
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    env_kwargs: dict = None,
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
):
    assert not (save_render and not render), "To save the video you have to render it"
    if env_kwargs is None:
        env_kwargs = {}

    # Scenario specific variables
    policy = heuristic(continuous_action=True)

    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0
    episode_cnt = 1
    for _ in range(n_steps):
        step += 1
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
        obs, rews, dones, info = env.step(actions)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )
        if all(dones):
            episode_cnt += 1
            obs = env.reset()

    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)
        

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {total_reward} with {episode_cnt} episodes\n"
        f"The episode reward was {total_reward / episode_cnt}"
    )


if __name__ == "__main__":
    from vmas.scenarios.transport import HeuristicPolicy as TransportHeuristic
    from vmas.scenarios.balance import HeuristicPolicy as BalanceHeuristic
    from vmas.scenarios.wheel import HeuristicPolicy as WheelHeuristic

    # In the training setting, there are 5 workers and 24 environments per worker, so 120 environments in total
    # The training steps are 500, so one iteration has 500 * 120 = 60000 steps
    # But in the heuristic setting, we only have one worker and 120 environments
    # So we need to run 500 steps to have the same amount of steps as in the training setting
    run_heuristic(
        scenario_name="transport",
        heuristic=TransportHeuristic,
        n_envs=24,
        n_steps=500,
        render=True,
        save_render=False,
    )