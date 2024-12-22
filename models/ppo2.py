import torch
from torch import nn
from ray.rllib.models import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimConv2d
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as CentralisedCritic

from utils import get_activation_fn

class PPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **cfg):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.encoder_out_features = 8
        self.shared_nn_out_features_per_agent = 8
        self.value_state_encoder_cnn_out_features = 16
        
        self.n_agents = len(obs_space.original_space.spaces) 
        self.outputs_per_agent = int(num_outputs / self.n_agents)
        self.obs_shape = obs_space.original_space.spaces[0].shape
        
        ########### Action NN ###########
        self.action_encoder = nn.Sequential(
            nn.Linear(self.obs_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.encoder_out_features),
            nn.ReLU(),
        )
        share_n_agents = 1
        self.action_shared = nn.Sequential(
            nn.Linear(self.encoder_out_features * share_n_agents, 64),
            nn.ReLU(),
            nn.Linear(64, self.shared_nn_out_features_per_agent * share_n_agents),
            nn.ReLU(),
        )
        post_logits = [
            nn.Linear(self.shared_nn_out_features_per_agent, 32),
            nn.ReLU(),
            nn.Linear(32, self.outputs_per_agent),
        ]
        nn.init.xavier_uniform_(post_logits[-1].weight)
        nn.init.constant_(post_logits[-1].bias, 0)
        self.action_output = nn.Sequential(*post_logits)
        
        ########### Value NN ###########
        self.value_encoder = nn.Sequential(
            nn.Linear(self.obs_shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, self.encoder_out_features),
            nn.ReLU(),
        )
        self.value_encoder_state = nn.Sequential(
            SlimConv2d(
                2, 8, 3, 2, 1
            ),  # in_channels, out_channels, kernel, stride, padding
            SlimConv2d(
                8, 8, 3, 2, 1
            ),  # in_channels, out_channels, kernel, stride, padding
            SlimConv2d(8, self.value_state_encoder_cnn_out_features, 3, 2, 1),
            nn.Flatten(1, -1),
        )
        self.value_shared = nn.Sequential(
            nn.Linear(
                self.encoder_out_features * self.n_agents,
                64,
            ),
            nn.ReLU(),
            nn.Linear(64, self.shared_nn_out_features_per_agent * self.n_agents),
            nn.ReLU(),
        )
        value_post_logits = [
            nn.Linear(self.shared_nn_out_features_per_agent, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        ]
        nn.init.xavier_uniform_(value_post_logits[-1].weight)
        nn.init.constant_(value_post_logits[-1].bias, 0)
        self.value_output = nn.Sequential(*value_post_logits)
        
        self._cur_value = None
        
    
    @override(ModelV2) 
    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"]: list, len=n_agents, each element is a tensor of shape (batch_size, obs_shape)
        batch_size = input_dict["obs"][0].shape[0]
        device = input_dict["obs"][0].device
        # print("input_dict['obs']", input_dict) #['obs', 'new_obs', 'actions', 'prev_actions', 'rewards', 'prev_rewards', 'dones', 'infos', 't', 'eps_id', 'unroll_id', 'agent_index', 'obs_flat']
        action_feature_map = torch.zeros(
            batch_size, self.n_agents, self.encoder_out_features
        ).to(device)
        value_feature_map = torch.zeros(
            batch_size, self.n_agents, self.encoder_out_features
        ).to(device)
        for i in range(self.n_agents):
            agent_obs = input_dict["obs"][i]
            action_feature_map[:, i] = self.action_encoder(agent_obs)
            value_feature_map[:, i] = self.value_encoder(agent_obs)
        value_state_features = self.value_encoder_state(
            input_dict["obs"][0].permute(0, 3, 1, 2)
        )
        
        action_shared_features = torch.empty(
            batch_size, self.n_agents, self.shared_nn_out_features_per_agent
        ).to(device)
        for i in range(self.n_agents):
            action_shared_features[:, i] = self.action_shared(
                action_feature_map[:, i]
            )
            
        value_shared_features = self.value_shared(
            torch.cat(
                [
                    value_feature_map.view(
                        batch_size, self.n_agents * self.encoder_out_features
                    ),
                    value_state_features,
                ],
                dim=1,
            )
        ).view(batch_size, self.n_agents, self.shared_nn_out_features_per_agent)
        
        outputs = torch.empty(batch_size, self.n_agents, self.outputs_per_agent).to(
            device
        )
        values = torch.empty(batch_size, self.n_agents).to(device)
        
        for i in range(self.n_agents):
            outputs[:, i] = self.action_output(action_shared_features[:, i])
            values[:, i] = self.value_output(value_shared_features[:, i]).squeeze(1)
        
        self._cur_value = values
        
        return outputs.view(batch_size, self.n_agents * self.outputs_per_agent), state
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
        
        
        