import torch
from jit_env import RotatorEnvironmentJit
from typing import NamedTuple

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor




class RolloutBufferGPU:
    def __init__(
        self,
        buffer_size: int, 
        env: RotatorEnvironmentJit, 
        device: torch.device = torch.device("cuda"),
        gae_lambda: float = 1, 
        gamma: float = 0.99
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.env = env
        self.n_agents = env.n_agents

        # Create actual fields
        self.observations = torch.zeros(
            (buffer_size * self.n_agents, env.n_observations_per_agent),
            device = self.device,
            dtype = torch.float32
        )
        self.actions = torch.zeros(
            (buffer_size * self.n_agents, env.n_action_parameters_per_agent),
            device = self.device,
            dtype = torch.float32
        )
        self.values = torch.zeros(
            (buffer_size * self.n_agents, 1), 
            device=self.device, 
            dtype=torch.float32
        )
        self.log_probs = torch.zeros(
            (buffer_size * self.n_agents, 1), 
            device=self.device, 
            dtype=torch.float32
        )

        # Fields that are more broken
        self.episode_starts = torch.zeros(
            (buffer_size, 1), # yea so like all agents "finish" at the same time
            device=self.device, 
            dtype=torch.float32
        )
        self.returns = torch.zeros(
            (self.buffer_size, self.n_agents), 
            device=self.device, 
            dtype=torch.float32
        )

        # these two interaction with eachother
        # we only have global reward
        self.rewards = torch.zeros((self.buffer_size, 1), device=self.device, dtype=torch.float32)
        self.advantages = torch.zeros(
            (self.buffer_size, self.n_agents),
            device=self.device,
            dtype=torch.float32
        )
        self.pos = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        episode_start: int, #maybe not this one
        value: torch.Tensor,
        log_prob: torch.Tensor
    ) -> None:
        
        self.observations[self.pos * self.n_agents: (self.pos + 1) * self.n_agents] = obs.clone()
        self.actions[self.pos * self.n_agents: (self.pos + 1) * self.n_agents]  = action.clone()
        self.values[self.pos * self.n_agents: (self.pos + 1) * self.n_agents] = value.clone()
        self.log_probs[self.pos * self.n_agents: (self.pos + 1) * self.n_agents]  = log_prob.clone()
        
        
        self.rewards[self.pos * self.n_agents: (self.pos + 1) * self.n_agents]  = reward
        self.episode_starts[self.pos] = episode_start
        
    
        self.pos += 1

    def compute_returns_and_advantage(self, last_values: torch.Tensor, done: int) -> None:
        # (n_agents,1) -> n_agents
        last_values = last_values.clone().flatten()
        last_gae_lam = 0
        
        self.values = self.values.view(self.buffer_size, self.n_agents)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1 - done
                next_values =  last_values
            else:
                next_non_terminal = 1 - self.episode_starts[step+1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.values + self.advantages

        # Fix all shapes
        self.returns = self.returns.view(self.buffer_size * self.n_agents, 1)
        self.advantages = self.advantages.view(self.buffer_size * self.n_agents, 1)

        # reshape values
        self.values = self.values.view(self.buffer_size * self.n_agents, 1)
        

    def reset(self):
        self.pos = 0
        self.returns = self.returns.view(self.buffer_size, self.n_agents)
        self.advantages = self.advantages.view(self.buffer_size, self.n_agents)
    
    def get(self, batch_size: int):
        # I'm not permuting anything rn 
        # this will effect accuracy or whatever but doesn't matter otherwise
        start_idx = 0
        while start_idx < self.buffer_size * self.n_agents:
            yield RolloutBufferSamples(
                self.observations[start_idx: start_idx + batch_size], #observations,
                self.actions[start_idx: start_idx + batch_size], #actions,
                self.values[start_idx: start_idx + batch_size].flatten(), #old_values,
                self.log_probs[start_idx: start_idx + batch_size].flatten(), #old_log_prob,
                self.advantages[start_idx: start_idx + batch_size].flatten(), #advantages,
                self.returns[start_idx: start_idx + batch_size].flatten(), #returns,
            )
            start_idx += batch_size
    