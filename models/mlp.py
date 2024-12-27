import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class Expert(nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=64):
        super().__init__()
        # 状态+动作的输入大小
        self.n_agents = len(action_space) # 4
        self.obs_shape = obs_space.original_space[0].shape[0] # 11
        self.act_shape = action_space[0].shape[0] # 2
        self.in_features = (self.obs_shape + self.act_shape) * self.n_agents # 52
        self.fc = nn.Sequential(
            nn.Linear(self.in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to("cuda")

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1).to("cuda")  # 拼接状态和动作
        return self.fc(x)
    
    def compute_loss(self, expert_preds, agent_preds):
        expert_labels = torch.ones_like(expert_preds)
        agent_labels = torch.zeros_like(agent_preds)
        
        expert_loss = F.binary_cross_entropy(expert_preds, expert_labels)
        agent_loss = F.binary_cross_entropy(agent_preds, agent_labels)
        
        total_loss = expert_loss + agent_loss
        return total_loss
    
    def predict_reward(self, obs, action):
        with torch.no_grad():
            mix = torch.cat([obs, action], dim=-1)
            out = self.forward(mix).squeeze()
            score = torch.sigmoid(out)
            reward = score.log() - (1-score).log()
        return reward