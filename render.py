import cv2
import torch
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

def render(algorithm, checkpoint_path, n_episodes):
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
    algorithm = "MAPPO"
    checkpoint_path = "C:\\Coding\\Python\\vmas_test\\reproduce\\ray_results\\transport\\MAPPO\\MultiPPOTrainer_transport_ebbd6_00000_0_2024-12-22_00-16-23\\checkpoint_000395\\policies\\default_policy"
    # export(checkpoint_path)
    render(algorithm, checkpoint_path, n_episodes=2)
