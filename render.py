'''
A script for rendering a video from a checkpoint.
Can be used to visualize the performance of the trained model.
python render.py --algorithm MAPPO --n_episodes 1
'''
import cv2
import torch
import argparse
from pathlib import Path

from utils import (
    get_config_trainer_and_env_from_checkpoint,
    rollout_episodes,
)
    
def export(checkpoint_path):
    # TODO: has to be tested
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

def render(algorithm, checkpoint_path, n_episodes, save_render=False):
    config, trainer, env = get_config_trainer_and_env_from_checkpoint(
        algorithm,
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
    
    if save_render:
        name = str(Path(checkpoint_path) / "render")
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
        
        print("The video has been saved to", f"{name}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VMAS Reproduce Render')
    parser.add_argument('--algorithm', type=str, default="CPPO")
    parser.add_argument('--n_episodes', type=int, default=7)
    parser.add_argument('--save_render', action="store_true", default=True)
    args = parser.parse_args()
    
    if args.algorithm == "CPPO":
        checkpoint_path = "ray_results\\transport\\CPPO\\PPO_transport_bc00e_00000_0_2024-12-19_22-15-44\\checkpoint_000392\\policies\\default_policy"
    elif args.algorithm == "MAPPO":
        checkpoint_path = "ray_results\\transport\\MAPPO\\MultiPPOTrainer_transport_ebbd6_00000_0_2024-12-22_00-16-23\\checkpoint_000395\\policies\\default_policy"
    else:
        checkpoint_path = "ray_results\\transport\\IPPO\\MultiPPOTrainer_transport_2466a_00000_0_2024-12-22_03-02-37\\checkpoint_000234\\policies\\default_policy"
    
    render(args.algorithm, checkpoint_path, n_episodes=args.n_episodes, save_render=args.save_render)
    # export(checkpoint_path)
