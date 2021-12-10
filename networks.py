import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        """Advantage Actor Critic Agent
        Args:
            input_dim (int): integer representing observation/state size
            n_actions (int): integer representing size of action space
        """
        super(ActorCritic, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_dim[0], out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        conv_out_size = self._get_conv_out(input_dim)
        
        # actor
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_actions)
        )

        # critic
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.initailize_weights()

    def _get_conv_out(self, shape):
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, observation):
        observation = observation.float()
        conv_out = self.features(observation.detach()).view(observation.size()[0], -1)
        
        # actor: find action prob_distribution
        policy = self.policy(conv_out)
        action_dist = F.softmax(policy, dim=1)

        # critic: find state-value
        value = self.value(conv_out)

        return value, action_dist

    def initailize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)