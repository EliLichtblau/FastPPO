import torch
from typing import Type, Optional, Tuple, Any, List, Dict

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import NatureCNN # prolly gonna have to reimplement this for multi-env
from stable_baselines3.common.torch_layers import MlpExtractor

from utils import *
from functools import partial

from env import Box


class ActorCriticPolicy(torch.nn.Module):
    def __init__(
        self,
        observation_space: Box, # only support continuos actions 
        action_space: Box, # only support continuos actions,
        lr_schedule: Schedule, # fuck function factories on god
        net_arch: Optional[Any] = None, # not sure what this is yet
        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None, # unsure what tf this is
        use_expln: bool = False, # no clue
        squash_output: bool = False, # no clue
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, # TODO: this NEEDS to be implemented
        features_extractor_kwargs: Optional[Dict[str, Any]] = None, #TODO: also needs to be handled probably
        # normalize images... we don't have images
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None, # not sure about this
    ):
        assert optimizer_kwargs is None, "Optimizer kwargs is not None and is not implemented"
        assert optimizer_class == torch.optim.Adam, "Only supports optimizer_class == torch.optim.Adam"
        self.device = torch.device("cuda")
        super(ActorCriticPolicy, self).__init__()
        # Possible should be self
        optimizer_kwargs = {}
        self.features_extractor_kwargs = {}
        if optimizer_class == torch.optim.Adam:
            optimizer_kwargs["eps"] = 1e-5


        # NOTE: this is basically a substitue super call
        self._init_base_model_fields(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            # squash_output=squash_output, this is done in BasePolicy not in base model and I'm not sure i need it
        )
        

        # Default network architecture, from stable-baselines
        # TODO: Handle net arch bull shit
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    
        
        # Set some passed parameters
        self.net_arch = net_arch # this needs to be set 
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # TODO: implement this bullshit.. its a passed parameter so like it'll fail in func w ays
        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert use_sde == False, "Use_sde options not yet implemented"
        self.use_sde = use_sde # this is sorta redundant since we do not support this

        self.action_dist = make_proba_distribution(action_space) # we can drop parameters because "use_sde" is always false

        # This prolly needs to be something that is
        self._build(lr_schedule)


    def _init_base_model_fields(
        self,
        observation_space: Box,
        action_space: Box,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[torch.nn.Module] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Basically the init of the BaseModel than actor critic inherits from
        TODO: delete stupid extra bull shit that shouldn't be being set and is confusing
        """
        # Honestly why don't these default to dictionaries lololol
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}


        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        
        
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None # This is prlly stupid but what ever



        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs # don't think this does shit


    def _get_constructor_parameters(self) -> None:
        raise Exception("_get_constructor_parameters(self) is not implemented and I don't think it should be")
    
    def reset_noise(self, n_envs: int = 1):
        raise Exception("SB3 only implemneted this for StateDependentNoiseDistribution")

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch, 
            activation_fn=self.activation_fn,
            device=self.device,
        )
    
    def _build(self, lr_schedule: Any) -> None:
        """
        Copied from sb3.common.policies but simpler because not accounting for multiple distributions
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        assert isinstance(self.action_dist, DiagGaussianDistribution), "Only supporting DiagGaussianDistrbution from action_distribution"

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )
        self.action_net = self.action_net.to(self.device)
        #self.log_std = self.log_std.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)
        #breakpoint()
        print(f"self.log_std.is_cuda: {self.log_std.is_cuda}")
        self.value_net = torch.nn.Linear(self.mlp_extractor.latent_dim_vf, 1, device=self.device)
        #self.value_net.to(self.device)
        #breakpoint()
        assert self.ortho_init == True, "I'm pretty sure not using orthogonal initialization is aggresively retarded"

        # NOTE: Copied sb3 
        # Honestly not sure what tf this is
        module_gains = {
            self.features_extractor: np.sqrt(2),
            self.mlp_extractor: np.sqrt(2),
            self.action_net: 0.01,
            self.value_net: 1,
        }

        for module, gain in module_gains.items():
            module.apply(partial(self.init_weights, gain=gain))

        # END random copied BS

        # Setup optimizer... I think self.parameters is a torch nn module method
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actor AND Critic forward pass
        :param obs: environment observastion
        :param deterministic: whether to sample or use deterministic actions?? not sure really sure atm
        return action, value, log_prob(action)
        """
        # Not sure if observation actually needs to be preprocessed
        features = self.extract_features(obs)
        
        #breakpoint()
        latent_pi, latent_vf = self.mlp_extractor(features)
        #breakpoint()
        # Evaluate the values for the given observations... y'know critic
        values = self.value_net(latent_vf)
        #breakpoint()
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob



    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
            Some magical preprocessing... god i love magic
        """
        assert self.features_extractor is not None, "self.features_extractor cannot be none lol"
        #preprocessed_obs = preprocess_obs(obs, self.observation_space)
        return self.features_extractor(obs)
    

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> DiagGaussianDistribution: # Distribuiton
        mean_actions = self.action_net(latent_pi)
        assert isinstance(self.action_dist, DiagGaussianDistribution), "self.action_dist only implemented for DiagGaussianDistribution"
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
    

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        #breakpoint()
        latent_pi, latent_vf = self.mlp_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
    

    def get_distribution(self, obs: torch.Tensor) -> DiagGaussianDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)


    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)



    @staticmethod
    def init_weights(module: torch.nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)


    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)




if __name__ == "__main__":
    from env import MultiRotatorEnvironment
    from stable_baselines3.common.utils import get_schedule_fn
    env = MultiRotatorEnvironment()
    lr_schedule = get_schedule_fn(0.1)
    #scripted_policy = torch.jit.script(ActorCriticPolicy(env.observation_space, env.action_space, lr_schedule))
