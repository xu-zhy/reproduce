#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from typing import Sequence

import gym
import numpy as np
from ray.rllib.models import ModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()


class MyFullyConnectedNetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **cfg,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.n_agents = len(obs_space.original_space)
        self.outputs_per_agent = num_outputs // self.n_agents
        self.trainer = cfg["trainer"]
        self.heterogeneous = cfg["heterogeneous"]
        self.pos_dim = cfg["pos_dim"]
        self.pos_start = cfg["pos_start"]
        self.vel_start = cfg["vel_start"]
        self.vel_dim = cfg["vel_dim"]
        self.use_beta = cfg["use_beta"]
        self.centralised_critic = cfg["centralised_critic"]
        self.use_beta = False

        self.obs_shape = obs_space.original_space[0].shape[0]
        # Remove position
        self.obs_shape -= self.pos_dim
        # print("obs_shape", self.obs_shape) #9
        if self.centralised_critic:
            self.share_obs_shape = self.obs_shape * self.n_agents
            
        self.actor_networks = nn.ModuleList(
            [
                MyFullyConnectedNetworkInner(
                    self.obs_shape,
                    self.outputs_per_agent,
                    self.n_agents,
                )
            ]
        )
        self.value_networks = nn.ModuleList(
            [
                MyFullyConnectedNetworkInner(
                    self.obs_shape if not self.centralised_critic else self.share_obs_shape,
                    1,
                    self.n_agents,
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
        
        if self.centralised_critic:
            share_obs = obs.reshape(batch_size, -1).cpu().numpy()
            share_obs = np.expand_dims(share_obs, 1).repeat(self.n_agents, axis=1)
            # to torch
            share_obs = torch.tensor(share_obs, dtype=torch.float32).to(device)
            
        # print("share_obs", obs.shape) #torch.Size([32, 4, 36])

        logits, _ = self.actor_networks[0](obs, state)
        if self.centralised_critic:
            value, _ = self.value_networks[0](share_obs, state)
        else:
            value, _ = self.value_networks[0](obs, state)

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


# class MyFullyConnectedNetworkInner(nn.Module):
#     """Generic fully connected network."""

#     def __init__(
#         self,
#         obs_shape: Sequence[int],
#         num_outputs: int,
#         model_config: ModelConfigDict,
#     ):
#         nn.Module.__init__(self)

#         hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
#             model_config.get("post_fcnet_hiddens", [])
#         )
#         activation = model_config.get("fcnet_activation")
#         if not model_config.get("fcnet_hiddens", []):
#             activation = model_config.get("post_fcnet_activation")
#         no_final_linear = model_config.get("no_final_linear")
#         self.vf_share_layers = model_config.get("vf_share_layers")
#         self.free_log_std = model_config.get("free_log_std")
#         # Generate free-floating bias variables for the second half of
#         # the outputs.
#         if self.free_log_std:
#             assert num_outputs % 2 == 0, (
#                 "num_outputs must be divisible by two",
#                 num_outputs,
#             )
#             num_outputs = num_outputs // 2

#         layers = []
#         prev_layer_size = int(np.product(obs_shape))
#         # print("prev_layer_size", prev_layer_size) #36
#         self._logits = None

#         # Create layers 0 to second-last.
#         for size in hiddens[:-1]:
#             layers.append(
#                 SlimFC(
#                     in_size=prev_layer_size,
#                     out_size=size,
#                     initializer=normc_initializer(1.0),
#                     activation_fn=activation,
#                 )
#             )
#             prev_layer_size = size

#         # The last layer is adjusted to be of size num_outputs, but it's a
#         # layer with activation.
#         if no_final_linear and num_outputs:
#             layers.append(
#                 SlimFC(
#                     in_size=prev_layer_size,
#                     out_size=num_outputs,
#                     initializer=normc_initializer(1.0),
#                     activation_fn=activation,
#                 )
#             )
#             prev_layer_size = num_outputs
#         # Finish the layers with the provided sizes (`hiddens`), plus -
#         # iff num_outputs > 0 - a last linear layer of size num_outputs.
#         else:
#             if len(hiddens) > 0:
#                 layers.append(
#                     SlimFC(
#                         in_size=prev_layer_size,
#                         out_size=hiddens[-1],
#                         initializer=normc_initializer(1.0),
#                         activation_fn=activation,
#                     )
#                 )
#                 prev_layer_size = hiddens[-1]
#             if num_outputs:
#                 self._logits = SlimFC(
#                     in_size=prev_layer_size,
#                     out_size=num_outputs,
#                     initializer=normc_initializer(0.01),
#                     activation_fn=None,
#                 )
#             else:
#                 self.num_outputs = ([int(np.product(obs_shape))] + hiddens[-1:])[-1]

#         # Layer to add the log std vars to the state-dependent means.
#         if self.free_log_std and self._logits:
#             self._append_free_log_std = AppendBiasLayer(num_outputs)

#         self._hidden_layers = nn.Sequential(*layers)

#         self._value_branch_separate = None
#         if not self.vf_share_layers:
#             # Build a parallel set of hidden layers for the value net.
#             prev_vf_layer_size = int(np.product(obs_shape))
#             vf_layers = []
#             for size in hiddens:
#                 vf_layers.append(
#                     SlimFC(
#                         in_size=prev_vf_layer_size,
#                         out_size=size,
#                         activation_fn=activation,
#                         initializer=normc_initializer(1.0),
#                     )
#                 )
#                 prev_vf_layer_size = size
#             self._value_branch_separate = nn.Sequential(*vf_layers)

#         self._value_branch = SlimFC(
#             in_size=prev_layer_size,
#             out_size=1,
#             initializer=normc_initializer(0.01),
#             activation_fn=None,
#         )
#         # Holds the current "base" output (before logits layer).
#         self._features = None
#         # Holds the last input, in case value branch is separate.
#         self._last_flat_in = None

#     @override(TorchModelV2)
#     def forward(self, obs, state,):
#         self._last_flat_in = obs
#         self._features = self._hidden_layers(self._last_flat_in)
#         logits = self._logits(self._features) if self._logits else self._features
#         if self.free_log_std:
#             logits = self._append_free_log_std(logits)
#         return logits, state

#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         if self._value_branch_separate:
#             return self._value_branch(
#                 self._value_branch_separate(self._last_flat_in)
#             ).squeeze(-1)
#         else:
#             return self._value_branch(self._features).squeeze(-1)
        
class MyFullyConnectedNetworkInner(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_agents,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.in_features = in_features
        self.out_features = out_features
        
        self.activation_fn = nn.Tanh
        
        self.fc1 = nn.Sequential(
            nn.Linear(
                self.in_features,
                512,
            ),
            self.activation_fn(),
            # nn.LayerNorm(256),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(
                512,
                256,
            ),
            self.activation_fn(),
            # nn.LayerNorm(self.out_features),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(
                256,
                self.out_features
            )
        )
    
    @override(TorchModelV2)
    def forward(self, obs, state):
        x = self.fc1(obs)
        x = self.fc2(x)
        x = self.fc3(x)
        return x, state
        