import torch
from gym import spaces
from stable_baselines3.common.distributions import DiagGaussianDistribution
from functools import reduce
import operator as op
import numpy as np
from typing import Tuple


"""
Yea so box has things I might need to implement but for now this fine
"""

class Box:
    def __init__(self, low: torch.Tensor, high: torch.Tensor, dtype: torch.dtype):
        self.low = low
        self.high = high
        self.dtype = dtype
        assert self.low.shape == self.high.shape, "Box low and high shapes must be the same"
    
    @property
    def shape(self):
        return self.low.shape



def make_proba_distribution(action_space: spaces.Box) -> DiagGaussianDistribution:
    """
    We only support continuous spaces, which is the spaces.Box from gym Env

    # NOTE: default DiagGaussianDistribution might be stupid worth checking 
    """
    return DiagGaussianDistribution(get_action_dim(action_space))

def get_action_dim(action_space: spaces.Box):
    return int(np.prod(action_space.shape))


def preprocess_obs(
    obs: torch.Tensor,
    observataion_space: spaces.Box, # only support continuous environments
) -> torch.Tensor:
    assert isinstance(observataion_space, spaces.Box), "Only support continuous actions... observation space"
    return obs.float() # no fucking clue what the fuck this is




class BaseFeaturesExtractor(torch.nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: Box, features_dim: int = 0):
        super(BaseFeaturesExtractor, self).__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: Box):
        super(FlattenExtractor, self).__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = torch.nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)



def get_flattened_obs_dim(space: Box) -> int:
    """
    Overwrite bullshit gym
    """
    #print(space.shape, type(space.shape))
    return reduce(op.mul, space.shape, 1)




def get_obs_shape(
    observation_space: Box,
) -> Tuple[int, ...]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    assert isinstance(observation_space, Box), "Passed observation space is not a Eli Box lol"
    return observation_space.shape



def get_action_dim(action_space: Box) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    assert isinstance(action_space, Box), "Passed action space is not a Eli box"
    
    return int(np.prod(action_space.shape))
    