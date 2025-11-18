# DQN implementation for Atari games with experience replay and target network

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
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, gamma=0.99, alpha=0.00025, epsilon=0.1, epsilon_decay_steps=500000, min_epsilon=0.01, beta=0.4, target_update_freq=1000, buffer_size=100000, batch_size=64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0
        self.actions = range(env.action_space.n)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        self.q_network = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=alpha)
        self.loss_fn = nn.HuberLoss()

        self.buffer_size = buffer_size  # N
        self.batch_size = batch_size
        self.alpha = 0.6  # controls the amount of prioritization
        self.beta = beta  # controls the amount of importance sampling correction
        storage = LazyTensorStorage(max_size=self.buffer_size)
        self.sampler = PrioritizedSampler(max_capacity=self.buffer_size, alpha=self.alpha, beta=self.beta)
        self.replay_buffer = ReplayBuffer(storage=storage, batch_size=self.batch_size, sampler=self.sampler, prefetch=4)
        
        self.target_network = DQN(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.target_update_counter = 0
        self.target_update_freq = target_update_freq

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
        q_value = q_values.gather(1, actions)  
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)  
            next_q_target = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_target

        td_errors = (q_value - target_q).detach().abs().squeeze()
        
        self.replay_buffer.update_priority(index, td_errors)

        loss = (self.loss_fn(q_value, target_q) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.target_update_counter += 1
        self.sampler.beta = min(1.0, self.sampler.beta + 1/ self.epsilon_decay_steps)
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        self.steps_done += 1
        if self.steps_done < self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.steps_done / self.epsilon_decay_steps)
        else:
            self.epsilon = self.epsilon_end

