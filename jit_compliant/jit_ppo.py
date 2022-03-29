import torch
from typing import Optional
from jit_actor_critic import ActorCriticPolicy
from jit_env import RotatorEnvironmentJit
import time
from jit_buffers import RolloutBufferGPU
import math
from torch.nn import functional as F

from stable_baselines3.common.utils import get_schedule_fn
from typing import Any
class PPO:
    def __init__(
        self,
        policy: ActorCriticPolicy,
        learning_rate: float=3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2, # they have an option for schedule... no fucking clue
        clip_range_vf: float = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        seed: Optional[int] = None,
        ):
            self.current_timestep = 0 # TODO: remove this is a placeholder
            self.device = torch.device('cuda')
            self.policy = policy
            self.env = actor.environment
            # NOTE: Called __init__ OnPolicyAlgorithm
            self._init_on_policy(
                n_steps,
                gamma,
                gae_lambda,
                ent_coef,
                vf_coef,
                max_grad_norm
            )

            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

            # Set passed parameters
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.clip_range = clip_range
            self.clip_range_vf = clip_range_vf
            self.learning_rate = learning_rate
            self._setup_model()
            self._setup_model_on_policy()
            # I'm putting this here as a sanity check... creating the data requires an initialization which
            # is actually done in learn but like i don't know.. I'm putting this here... kill me
            self._last_obs = self.env.reset()
            self._last_episode_starts = 1

            # TODO: Implement rollout buffer on gpu memory
            # Allocate rollout buffer in gpu memory...
            # yay I get to think about sizing uhhhh
            
            # self.rollout_buffer = torch.cuda.FloatTensor()

            # so we will spawn n environments such that stepping through all of them executes 
            self.n_environments = math.ceil(float(self.n_steps) / self.env.n_agents / self.env.max_cycles)
            print(self.n_environments)
        
    def _setup_model(self) -> None:
        # TODO: implement
        # Call super _setup_model
        '''
        class RolloutBufferGPU:
        def __init__(
            self,
            buffer_size: int,
            n_agents: int, # WHY DOES STABLEBASELINES CALL THIS N_ENVS FUCK ME
            single_observation_shape: Tuple[int, ...],
            single_action_shape: Tuple[int, ...],
            single_reward_shape: Tuple[int, ...],
            device: torch.device = torch.device("cuda"),
            gae_lambda: float = 1,
            gamma: float = 0.99,
            
        ):
        '''
        
        self.rollout_buffer = RolloutBufferGPU (
            self.n_steps,
            self.env,
        )
        #breakpoint()

        # Initiliaze schedule for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None: # Honest to god no fucking clue what this does
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)


    def _setup_model_on_policy(self) -> None:
        """
        Functions as super call, inheriting a shitton of methods is like really fucking bad for debugging
        """
        self._setup_lr_schedule()
        # NOTE: need to uncomment and implement at some point
        # self.set_random_seed(self.seed)

        # TODO: IMPLEMENT THIS FUNCTION
        self.policy.to(self.device)

    def _init_on_policy(
        self,
        n_steps,
        gamma,
        gae_lambda,
        ent_coef,
        vf_coef,
        max_grad_norm
    ):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None # This is garbage!!




    def _setup_lr_schedule(self):
        self.lr_schedule = get_schedule_fn(self.learning_rate)
    
    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizer: torch.optim.Optimizer):
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    def set_random_seed(self, seed: Any):
        raise Exception("need to implement lol, idk cuda environment bs kill me")
    
    def train(self) -> None:
        # TODO: implement
        # drop out bull shit? but like not implemented atm
        s = time.time()
        #self.policy.set_training_mode(True)
        self.policy.train(True)
        self._update_learning_rate(self.policy.optimizer)

        # NOTE: I actually don't know what clip range or clip_range_vf is
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)


        # store losses idk why honestly
        #entropy_losses = []
        #pg_losses, value_losses = [], []
        #clip_fractions = []

        # this is here... god knows why
        continue_training = True
        
        for epoch in range(self.n_epochs):
            #approx_kl_divs = []
            # Okay they call get on batchsize number of actions
            # so we will do that in a less retarded way
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions #flatten() ??
                observations = rollout_data.observations
                #breakpoint()
                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                # values = values.flatten() ???

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                #breakpoint()
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Just running with clip_range_vf is None
                values_pred = values.flatten() # I should have to do this
                #breakpoint()
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                #value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                #entropy_losses.append(entropy_loss.item())
                #breakpoint()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                '''
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)#.cpu().numpy()
                    #approx_kl_divs.append(approx_kl_div)
                
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                '''
                

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            


            if not continue_training:
                break
            
        print(f"training time: {time.time() - s}")

            
    def collect_rollouts2(self):
        """
        Gets self.n_steps experiences from current policy
        that is:
        observations, actions, rewards, values, log probs
        """
        assert self._last_obs is not None, "We need a previous observation to collect rollouts"
        
        s = time.time()
        self.policy.set_training_mode(False)
        #breakpoint()
        for i in range(self.n_steps):
            #print(f"collect_rollouts. step: {i}")
            with torch.no_grad():
                # we return a tensor already aint that nice 
                actions, values, log_probs = self.policy.forward(self._last_obs)
            
            # we know our shit is a box
            clipped_actions = torch.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            new_obs, global_reward, done = self.env.step(clipped_actions)
            #breakpoint()
            self.current_timestep += 5 # yea this is kinda dumb I think I should set this later whatever
            #observations: torch.Size([5, 12]) actions: torch.Size([5, 2]) rewards: torch.Size([5]) values: torch.Size([5, 1]) log_probs: torch.Size([5])
            
            # push to buffer
            self.rollout_buffer.add(self._last_obs, actions, global_reward, self._last_episode_starts, values, log_probs.view(5,1))
            self._last_obs = new_obs
            self._last_episode_starts = done

            
        with torch.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(new_obs)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, done=done)
        print(f"collect_rollouts: {time.time() - s}")

    def collect_rollouts(self):
        """
        Gets self.n_steps experiences from current policy
        that is:
        observations, actions, rewards, values, log probs
        """
        assert self._last_obs is not None, "We need a previous observation to collect rollouts"
        
        s = time.time()
        #self.policy.set_training_mode(False)
        self.policy.train(False)
        #breakpoint()
        for i in range(self.n_steps):
            #print(f"collect_rollouts. step: {i}")
            with torch.no_grad():
                # we return a tensor already aint that nice 
                actions, values, log_probs = self.policy.forward(self._last_obs)
            
            # we know our shit is a box
            clipped_actions = torch.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            new_obs, global_reward, done = self.env.step(clipped_actions)
            #breakpoint()
            self.current_timestep += 5 # yea this is kinda dumb I think I should set this later whatever
            #observations: torch.Size([5, 12]) actions: torch.Size([5, 2]) rewards: torch.Size([5]) values: torch.Size([5, 1]) log_probs: torch.Size([5])
            
            # push to buffer
            self.rollout_buffer.add(self._last_obs, actions, global_reward, self._last_episode_starts, values, log_probs.view(5,1))
            self._last_obs = new_obs
            self._last_episode_starts = done

            
        with torch.no_grad():
                # Compute value for the last timestep
                values = self.policy.predict_values(new_obs)

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, done=done)
        print(f"collect_rollouts: {time.time() - s}")











    def learn(
        self,
        total_timesteps: int,
        # not doing this callback: MaybeCallback = None,
        log_interval: int = 1,
        # I don't think I want to deal with this yet eval_env: Optional[GymEnv] = None,
        # eval_freq: int = -1,
        # n_eval_episodes: int = 5,
        reset_num_timesteps: bool = True,
    ):
        # the first thing that needs to be done is to set last_obs
        self._last_obs = self.env.reset()

        # TODO: implement
        iteration = 0
        # Pretty sure this useless... i mean it does timing bs but fuck that
        #total_timesteps, callback = self._setup_learn(
        #    total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        #)
        self.current_timestep = 0 # This is updated in n_envs and n_envs only...???

        while self.current_timestep < total_timesteps:
            #self.rollout_buffer.print()
            self.collect_rollouts()
            # code calls collect rollouts here
            iteration += 1
            # this just updates the learning rate
            self._update_current_progress_remaining(self.current_timestep, total_timesteps)

            # logging bullshit... skipping

            self.train()
            self.rollout_buffer.reset() # This is slow and stupid y'know
        return self



#env = torch.jit.script(RotatorEnvironmentJit())
lr_schedule = get_schedule_fn(0.1)
actor = torch.jit.script(ActorCriticPolicy(RotatorEnvironmentJit, lr_schedule))
p = PPO(actor)
print("Starting...")
s = time.time()
p.learn(int(2048 * 10))
print(f"full time: {time.time() - s}")