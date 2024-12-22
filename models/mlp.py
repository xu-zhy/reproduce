import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class MLP(nn.Module):
    def __init__(self, in_features, out_features, n_agents):
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
        