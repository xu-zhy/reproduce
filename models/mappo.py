import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from models.mlp import MLP

class MAPPO(TorchModelV2, nn.Module):
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **cfg,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_agents = len(obs_space.original_space)
        self.outputs_per_agent = num_outputs // self.n_agents
        self.trainer = cfg["trainer"]
        self.pos_dim = cfg["pos_dim"]
        self.pos_start = cfg["pos_start"]
        self.vel_start = cfg["vel_start"]
        self.vel_dim = cfg["vel_dim"]
        self.use_beta = cfg["use_beta"]
        self.add_noise = cfg["add_noise"]
        print("MAPPO: add_noise", self.add_noise)
        
        self.obs_shape = obs_space.original_space[0].shape[0]
        self.obs_shape -= self.pos_dim
        self.share_obs_shape = self.obs_shape * self.n_agents
        # print("obs_shape", self.obs_shape) #9
            
        self.actor_networks = nn.ModuleList(
            [
                MLP(
                    self.obs_shape,
                    self.outputs_per_agent,
                    self.n_agents,
                )
            ]
        )
        self.value_networks = nn.ModuleList(
            [
                MLP(
                    self.share_obs_shape, 
                    1,
                    self.n_agents,
                    self.add_noise
                )
            ]
        )
        self.share_init_hetero_networks()

    def forward(self, input_dict, state, seq_lens):
        batch_size = input_dict["obs"][0].shape[0]
        device = input_dict["obs"][0].device

        obs = torch.stack(input_dict["obs"], dim=1)
        obs_no_pos = torch.cat(
            [
                obs[..., : self.pos_start],
                obs[..., self.pos_start + self.pos_dim :],
            ],
            dim=-1,
        ).view(
            batch_size, self.n_agents, self.obs_shape
        )  # This acts like an assertion
        obs = obs_no_pos
        
        share_obs = obs.reshape(batch_size, -1).cpu().numpy()
        share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
        # to torch
        share_obs = torch.tensor(share_obs, dtype=torch.float32).to(device)
        # print("share_obs", obs.shape) #torch.Size([32, 4, 36])

        logits, _ = self.actor_networks[0](obs, state)
        value, _ = self.value_networks[0](share_obs, state)

        outputs = logits.view(batch_size, self.n_agents * self.outputs_per_agent)
        values = value.view(batch_size, self.n_agents)
        
        self._cur_value = values
        
        return outputs, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def share_init_hetero_networks(self):
        for child in self.children():
            assert isinstance(child, nn.ModuleList)
            for agent_index, agent_model in enumerate(child.children()):
                if agent_index == 0:
                    state_dict = agent_model.state_dict()
                else:
                    agent_model.load_state_dict(state_dict)