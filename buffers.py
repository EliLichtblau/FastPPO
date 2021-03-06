from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple, Tuple
from abc import ABC, abstractmethod
import numpy as np
import torch
from utils import Box, get_obs_shape, get_action_dim

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Box,
        action_space: Box,
        device: Union[torch.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False
    
    '''
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()
    
    '''
    

    def to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)


    '''
    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward
    
    '''
    




class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: Box,
        action_space: Box,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):

        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        #if isinstance(self.observation_space, spaces.Discrete):
        #    obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))





class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor



from env import RotatorEnvironment
class RolloutBufferGPU:
    def __init__(
        self,
        buffer_size: int, 
        env: RotatorEnvironment, 
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
    


