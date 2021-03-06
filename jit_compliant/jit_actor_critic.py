import torch
import numpy as np
from jit_env import RotatorEnvironmentJit, RotatorWorldJit
from functools import partial
from typing import Callable, Optional, Any, Type, Dict, Tuple
from jit_utils import get_schedule_fn, Schedule, BaseFeaturesExtractor, FlattenExtractor, make_proba_distribution
from jit_distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import NatureCNN # prolly gonna have to reimplement this for multi-env
from stable_baselines3.common.torch_layers import MlpExtractor

class ActorCriticPolicy(torch.nn.Module):
    def __init__(
        self,
        environment: RotatorEnvironmentJit,
        lr_schedule: Schedule,
        net_arch: Optional[Any] = None, # not sure what this is yet
        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor, # TODO: this NEEDS to be implemented
        features_extractor_kwargs: Optional[Dict[str, Any]] = None, #TODO: also needs to be handled probably
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam, # I honestly don't see this change
        optimizer_kwargs: Optional[Dict[str, Any]] = None
        
    ):
        super(ActorCriticPolicy, self).__init__()
        self.device = torch.device("cuda")
        self.lr_schedule = lr_schedule

        self.environment = environment # TODO: maybe make this more like accept a type?
        self.observation_space = self.environment.observation_space

        self.action_space = self.environment.action_space

        # features extractor... exists for MLP extractor
        self.features_extractor_class = features_extractor_class
        if features_extractor_kwargs == None:
            self.features_extractor_kwargs = {}
        else:
            self.features_extractor_kwargs = features_extractor_kwargs
        
        # initialize optimizer parameters
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        if self.optimizer_class == torch.optim.Adam:
            self.optimizer_kwargs["eps"] = 1e-5
        
        # Net arch parameters for mlp extractor
        self.net_arch = net_arch
        if self.net_arch is None:
            self.net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            if self.features_extractor_class == NatureCNN: # NatureCNN might break me code
                self.net_arch = []

        # Just some normal model parameters
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # use passed features extractor class to make a features extractor
        self.features_extractor = self.features_extractor_class(self.observation_space)#, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

         
        self.log_std_init = log_std_init
        assert use_sde == False, "Use_sde options not yet implemented"
        self.use_sde = use_sde # this is sorta redundant since we do not support this

        self.action_dist = make_proba_distribution(self.action_space) # NOTE: possible rewrite required... distribution is not torchscript...

        # Build MLP extractor
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

        # build function
        assert isinstance(self.action_dist, DiagGaussianDistribution), "Only supporting DiagGaussianDistrbution from action_distribution"
        self.action_net, self.log_std = self.action_dist.proba_distribution_net( 
                latent_dim=self.mlp_extractor.latent_dim_pi, log_std_init=self.log_std_init
        )

        self.action_net = self.action_net.to(self.device)
        self.log_std.data = self.log_std.data.to(self.device)

        self.value_net = torch.nn.Linear(self.mlp_extractor.latent_dim_vf, 1, device=self.device)

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
        # TODO: fix this bs
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1))#, **self.optimizer_kwargs)




    
    def forward(self, obs: torch.Tensor, deterministic: bool = False): #-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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



    @torch.jit.export
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









    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
            Some magical preprocessing... god i love magic
        """
        assert self.features_extractor is not None, "self.features_extractor cannot be none lol"
        #preprocessed_obs = preprocess_obs(obs, self.observation_space)
        return self.features_extractor(obs)



    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor): #-> DiagGaussianDistribution: # Distribuiton
        mean_actions: torch.Tensor = self.action_net(latent_pi)
        
        assert isinstance(self.action_dist, DiagGaussianDistribution), "self.action_dist only implemented for DiagGaussianDistribution"
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
        #return mean_actions

    def get_distribution(self, obs: torch.Tensor) -> DiagGaussianDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    @torch.jit.export
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


    #@torch.jit.export
    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


    @staticmethod
    def init_weights(module: torch.nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)




def collect_rollouts(actorcritic, last_obs):
    
    for i in range(2048):
        with torch.no_grad():
            # we return a tensor already aint that nice 
            actions, values, log_probs = actorcritic.forward(last_obs)
        clipped_actions = torch.clip(actions, actorcritic.environment.action_space.low, actorcritic.environment.action_space.high)
            
        #new_obs, global_reward, done = actorcritic.environment.step(actions)

        #last_obs = new_obs





if __name__ == "__main__":
    #env = torch.jit.script(RotatorEnvironmentJit())
    lr_schedule: Schedule = get_schedule_fn(0.1)
    #env = RotatorEnvironmentJit()
    #breakpoint()
    env = torch.jit.script(RotatorEnvironmentJit())
    #env = RotatorEnvironmentJit()
    actorcritic = ActorCriticPolicy(env, lr_schedule)
    #a = env.observation_space
    '''
    m = torch.jit.script(ActorCriticPolicy(env, lr_schedule)) #20k -> 5.1 seconds with jit
    #m = ActorCriticPolicy(RotatorEnvironmentJit, lr_schedule) #20k -> 8.5
    print("Starting")
    obs = m.environment.world.observation()
    actions, _, _ = m(obs)
    import time
    s = time.time()
    #for _ in range(20_000):
    #    m.evaluate_actions(obs, actions)
    collect_rollouts(m, obs)
    print(f"Time: {time.time() - s}")
    obs = m.environment.world.observation()
    
    
    '''
    




    breakpoint()