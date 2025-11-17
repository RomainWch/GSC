# DQN implementation with experience replay and without target network

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data import ReplayBuffer, LazyTensorStorage


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, gamma=0.99, alpha=0.00025, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.actions = range(env.action_space.n)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        self.buffer_size = 100000  # N
        self.batch_size = 64
        self.alpha = 0.6  # controls the amount of prioritization
        self.beta = 1  # controls the amount of importance sampling correction
        storage = LazyTensorStorage(max_size=self.buffer_size)
        self.sampler = PrioritizedSampler(max_capacity=self.buffer_size, alpha=self.alpha, beta=self.beta)
        self.replay_buffer = ReplayBuffer(storage=storage, batch_size=self.batch_size, sampler=self.sampler, prefetch = 1)

    def play(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor([action])
        next_state_tensor = torch.FloatTensor(next_state)
        reward_tensor = torch.FloatTensor([reward])
        done_tensor = torch.FloatTensor([done])

        self.replay_buffer.add((state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor))

        if len(self.replay_buffer) < self.batch_size:
            return

        batch, info = self.replay_buffer.sample(return_info=True)
        states, actions, rewards, next_states, dones = batch
        index = info["index"]
        weights = info['_weight']  
        
        q_values = self.q_network(states)  
        q_values_for_actions = q_values.gather(1, actions)  
        
        with torch.no_grad():
            max_next_q = self.q_network(next_states).max(dim=1, keepdim=True)[0]  
            target_q = rewards + (1 - dones) * self.gamma * max_next_q 

        td_errors = (q_values_for_actions - target_q).detach().abs().squeeze()
        
        self.replay_buffer.update_priority(index, td_errors)

        loss = (weights.unsqueeze(1) * (q_values_for_actions - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

