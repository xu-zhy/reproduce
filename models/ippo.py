import torch
from torch import nn
from ray.rllib.models import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as CentralisedCritic

from utils import get_activation_fn

class PPOBranch(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_agents,
        centralised,
        **cfg,
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.in_features = in_features
        self.out_features = out_features
        self.centralised = centralised
        
        self.activation_fn = get_activation_fn(cfg["activation_fn"])
        
        self.heterogeneous = cfg["heterogeneous"]
        
        if self.centralised:
            self.hidden_size = 128
            self.mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(
                            self.in_features * self.n_agents,
                            256,
                        ),
                        self.activation_fn(),
                        torch.nn.Linear(
                            256,
                            self.hidden_size * self.n_agents,
                        ),
                    )
                    for _ in range(self.n_agents if self.heterogeneous else 1)
                ]
            )
        else:
            self.hidden_size = 16
            self.mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(
                            self.in_features,
                            32,
                        ),
                        self.activation_fn(),
                        torch.nn.Linear(
                            32,
                            self.hidden_size,
                        ),
                    )
                    for _ in range(self.n_agents if self.heterogeneous else 1)
                ]
            )
        
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.hidden_size, self.hidden_size
                    ),
                    self.activation_fn(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    self.activation_fn(),
                )
                for _ in range(self.n_agents if self.heterogeneous else 1)
            ]
        )
        
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(self.hidden_size, self.out_features),
                )
                for _ in range(self.n_agents if self.heterogeneous else 1)
            ]
        )
        self.share_init_hetero_networks()
        
    def forward(self, obs, pos, vel):
        batch_size = obs.shape[0]
        device = obs.device
        if self.centralised:
            # for centralised critic, send all agents' observations to mlp
            # if heterogeneous, each agent has its own mlp
            embedding = obs.view(batch_size, self.n_agents * self.in_features)
            
            if self.heterogeneous:
                embedding = torch.stack(
                    [
                        mlp(embedding).view(
                            batch_size,
                            self.n_agents,
                            self.hidden_size,
                        )[:, i]
                        for i, mlp in enumerate(self.mlps)
                    ],
                    dim=1,
                )
            else:
                embedding = self.mlps[0](embedding).view(
                    batch_size,
                    self.n_agents,
                    self.hidden_size,
                )
        else:
            # for decentralised critic, send each agent's observation to its own mlp
            if self.heterogeneous:
                embedding = torch.stack(
                    [
                        mlp(obs[:, i]).view(
                            batch_size,
                            1,
                            self.hidden_size,
                        )
                        for i, mlp in enumerate(self.mlps)
                    ],
                    dim=1,
                )
            else:
                embedding = self.mlps[0](obs).view(
                    batch_size,
                    self.n_agents,
                    self.hidden_size,
                )
                
        if self.heterogeneous:
            embedding = torch.stack(
                [
                    decoder(torch.cat([obs[:, i], embedding[:, i]], dim=-1))
                    for i, decoder in enumerate(self.decoders)
                ],
                dim=1,
            )
        else:
            embedding = self.decoders[0](torch.cat([obs, embedding], dim=-1))
            
        if self.heterogeneous:
            logits = torch.stack(
                [head(embedding[:, i]) for i, head in enumerate(self.heads)],
                dim=1,
            )
        else:
            logits = self.heads[0](embedding)
        
        return logits
                
    def share_init_hetero_networks(self):
        for child in self.children():
            assert isinstance(child, nn.ModuleList)
            for agent_index, agent_model in enumerate(child.children()):
                if agent_index == 0:
                    state_dict = agent_model.state_dict()
                else:
                    agent_model.load_state_dict(state_dict)

class PPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **cfg):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.trainer = cfg["trainer"]                      
        self.pos_dim = cfg["pos_dim"]                        
        self.vel_dim = cfg["vel_dim"]                   
        self.pos_start = cfg["pos_start"]                     
        self.vel_start = cfg["vel_start"]   
        self.use_beta = cfg["use_beta"]
          
        self.use_mlp = cfg["use_mlp"]               # if true, use MLP         
        self.heterogeneous = cfg["heterogeneous"]   # if true, agents have different networks                            
        self.centralised_critic = cfg["centralised_critic"] # if true, use centralised critic
        
        if self.use_mlp:
            assert self.centralised_critic, "MLP only works with centralised critic"
        
        self.n_agents = len(obs_space.original_space.spaces)            # agent number
        self.output_per_agent = num_outputs // self.n_agents            # output per agent
        self.obs_shape = obs_space.original_space.spaces[0].shape[0]    # observation shape
        
        self.actor_networks = PPOBranch(
            in_features=self.obs_shape,
            out_features=self.output_per_agent,
            n_agents=self.n_agents,
            centralised=self.use_mlp,
            **cfg,
        )
        self.value_networks = PPOBranch(
            in_features=self.obs_shape,
            out_features=1,
            n_agents=self.n_agents,
            centralised=self.centralised_critic,
            **cfg,
        )
        self._cur_value = None
        
        # if self.centralised_critic:
        #     self.value_networks = CentralisedCritic(obs_space, action_space, num_outputs, model_config, name)
        # else:
        #     self.value_networks = DecentralisedCritic(obs_space, action_space, num_outputs, model_config, name)
    
    @override(ModelV2) 
    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"]: list, len=n_agents, each element is a tensor of shape (batch_size, obs_shape)
        batch_size = input_dict["obs"][0].shape[0]
        
        obs = torch.stack(input_dict["obs"], dim=1)
        obs_no_pos = torch.cat(
            [
                obs[..., : self.pos_start],
                obs[..., self.pos_start + self.pos_dim :],
            ],
            dim=-1,
        ).view(
            batch_size, self.n_agents, self.obs_shape
        )
        outputs = self.actor_networks(obs_no_pos, None, None)
        values  = self.value_networks(obs_no_pos, None, None)
        
        outputs = outputs.view(batch_size, self.n_agents * self.output_per_agent)
        values = values.view(batch_size, self.n_agents)
        # if self.trainer == "PPOTrainer":
        #     # assert self.n_agents == 1
        #     values = values.squeeze(-1)
        self._cur_value = values
        
        # outputs, _ = self.actor_networks(input_dict, state, seq_lens)
        # values = self.value_networks
        return outputs, state
    
    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value
        
        
        