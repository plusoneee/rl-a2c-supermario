import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from networks import ActorCritic
from memory import Memory
from tensorboardX import SummaryWriter

class Mario(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, discount_factor: float=0.99, learning_rate=0.0001):
        """Advantage Actor Critic Agent
        Args:
            input_dim (int): integer representing observation/state size
            n_actions (int): integer representing size of action space
            discount_factor (float, optional): [description]. Defaults to 0.99.
        """
        super(Mario, self).__init__()
        self.writer = SummaryWriter(log_dir='log/')
        self.gamma = discount_factor
        self.memory = Memory()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.model = ActorCritic(
            n_actions=n_actions,
            input_dim=input_dim
        ).to(self.device)

        self.entropy = 0
        self.ep_policy = torch.Tensor()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def choose_action(self, observation):

        if self.use_cuda:
            obs = torch.cuda.FloatTensor(observation.to(self.device))
        else:
            obs = torch.FloatTensor(observation)

        value, action_dist = self.model(obs)

        # pi(at|st) dist, categorical wrapper
        prob = Categorical(action_dist)

        # Sampling actions from policy distribution
        action = prob.sample()

        # store policy history
        self.ep_policy = (prob.log_prob(action))
        action = action.to('cpu').numpy()

        # entropy term for loss function, no backprop [constant]
        dist = action_dist.detach()
        entropy = -torch.sum(dist.mean() * torch.log(dist))
        self.entropy += entropy
        return value, action.item()

    def update(self, Q_value, episode):


        advantage = self._advantage(Q_value).to(self.device)
        actor_loss = (torch.sum(torch.mul(self.ep_policy, advantage).mul(-1), -1)).to(self.device)
        critic_loss = 0.5 * advantage.pow(2).mean().to(self.device)

        # total loss
        a2c_loss = actor_loss + critic_loss + 0.001 * self.entropy
        print(f'- Total A2C loss: {a2c_loss.item()}\n- Actor loss: {actor_loss.item()}\n- Critic loss: {critic_loss.item()}')

        self.writer.add_scalar('episode total loss', float(a2c_loss.item()), episode)
        self.writer.add_scalar('episode actor loss', float(actor_loss.item()), episode)
        self.writer.add_scalar('episode critic loss', float(critic_loss.item()), episode)
        self.writer.add_scalar('episode reward', float(sum(self.memory.ep_rewards)), episode)    

        self.optimizer.zero_grad()
        a2c_loss.backward(retain_graph=True)
        self.optimizer.step()

        # reset agent for next episode
        self.memory.clear()
        self.ep_policy = torch.Tensor()
        
        return advantage
    
    def _advantage(self, Q_value):
        Q_values = []
        values = []

        # reversed rewards
        for r in self.memory.ep_rewards[::-1]:
            # Q_(s_t, a_t)
            Q_value = r + self.gamma * Q_value
            Q_values.insert(0, Q_value)

        q_values = torch.FloatTensor(Q_values)
        
        # detach and store current state values according to time step
        for element in self.memory.ep_values:
            values.insert(0, element.detach())
        values = torch.FloatTensor(values)

        # Advantage = r_(t+1) + gamma* V_(s_t+1) - V_(s_t)
        advantage = q_values - values
        return advantage