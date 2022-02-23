import torch
from typing import Tuple, NamedTuple

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
        n_agents: int, # WHY DOES STABLEBASELINES CALL THIS N_ENVS FUCK ME
        n_observations_per_agent: int,
        n_action_per_agent: int,
        single_reward_shape: Tuple[int, ...],
        device: torch.device = torch.device("cuda"),
        gae_lambda: float = 1,
        gamma: float = 0.99,
        
    ):
        self.buffer_size = buffer_size
        
        
        self.n_observations_per_agent = n_observations_per_agent
        self.n_action_per_agent = n_action_per_agent
        self.reward_shape = single_reward_shape
        self.n_agents = n_agents

        #self.obs_shape = get_obs_shape(observation_space)

        #self.action_dim = get_action_dim(action_space)
        #self.n_envs = n_envs
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self._reset()
    
    def _reset(self) -> None:

        
        # Here is the current plan, we know the shapes of single observations atm, look like this
        # observations: torch.Size([5, 12]) actions: torch.Size([5, 2]) rewards: torch.Size([5]) values: torch.Size([5, 1]) log_probs: torch.Size([5])
        # so we allocate (buffer_size, *T.shape) where T is observations, actions, rewards.. respectively
        # buffer size is how many forward steps are execute before back pass
        self.observations = torch.zeros((int(self.buffer_size * self.n_agents), self.n_observations_per_agent), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((self.buffer_size * self.n_agents, self.n_action_per_agent), device=self.device, dtype=torch.float32)
        # TODO: possibly make this less stupid
        self.values = torch.zeros((self.buffer_size * self.n_agents, 1), device=self.device, dtype=torch.float32)
        self.log_probs = torch.zeros((self.buffer_size * self.n_agents, 1), device=self.device, dtype=torch.float32)
        # unsure of the sizing here
        
        # we neeed to fuck with you
        self.rewards = torch.zeros((self.buffer_size * self.n_agents, 1), device=self.device, dtype=torch.float32)
        
        self.advantages = torch.zeros((self.buffer_size, self.n_agents), device=self.device, dtype=torch.float32)


        # unsure about self.returns and episode starts
        self.returns = torch.zeros((self.buffer_size, self.n_agents), device=self.device, dtype=torch.float32)
        self.episode_starts = torch.zeros((self.buffer_size, 1), device=self.device, dtype=torch.float32)

        self.pos = 0 # for adding in shit
        #breakpoint()
        
    def reset(self):
        self.pos = 0
        self.returns = self.returns.view(self.buffer_size, self.n_agents)
        self.advantages = self.advantages.view(self.buffer_size, self.n_agents)

    def compute_returns_and_advantage(self, last_values: torch.Tensor, done: int):
        # TODO: rewrite this function to be less idk fucking retarded
        
        # all agents "finish" at same time
        last_values = last_values.clone().view(5)
        #breakpoint()
        last_gae_lam = 0

        # reshape for this operation
        self.rewards = self.rewards.view(self.buffer_size, self.n_agents)
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
        # reshape return cuz idk
        self.returns = self.returns.view(self.buffer_size * self.n_agents, 1)
        self.advantages = self.advantages.view(self.buffer_size * self.n_agents, 1)

        # reshape self.values back
        self.values = self.values.view(self.buffer_size*self.n_agents, 1)


        self.rewards = self.rewards.view(self.buffer_size * self.n_agents, 1)

        #breakpoint()

    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        episode_start: int, #maybe not this one
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        #breakpoint()
        #breakpoint()
        self.observations[self.pos * self.n_agents: (self.pos + 1) * self.n_agents] = obs.clone()
        #breakpoint()
        self.actions[self.pos * self.n_agents: (self.pos + 1) * self.n_agents]  = action.clone()
        self.rewards[self.pos * self.n_agents: (self.pos + 1) * self.n_agents]  = reward.clone()
        
        self.values[self.pos * self.n_agents: (self.pos + 1) * self.n_agents] = value.clone()
        self.log_probs[self.pos * self.n_agents: (self.pos + 1) * self.n_agents]  = log_prob.clone()
        #breakpoint()
        # unsure about what tf is going on with this guy
        self.episode_starts[self.pos] = episode_start

        self.pos += 1
    

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
    


    def print(self):    
        print(f"self.observations: {self.observations.shape}\nself.actions: {self.actions.shape}\nself.values: {self.values.shape}\nself.log_probs: {self.log_probs.shape}\nself.advantages: {self.advantages.shape}\nself.returns {self.returns.shape}")
        print()