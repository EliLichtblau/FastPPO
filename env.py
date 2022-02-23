import torch
import time
from typing import List
from gym import spaces # for action and observation spaces atm


"""
Implementation of Rotator World Environment that can support multiple environments
very useful for hypo tuning, basically just need to do the same with model lololol 
"""


class MultiEnvRotatorWorld:
    """
    TODO: Find some way of making pair distance faster?
    TODO: Write Cuda Layer if want ragged n_agents
    """
    def __init__(self, n_agents_per_environment: int, n_landmarks_per_environment: int, n_environments: int, use_gpu=True):
        # Set device first
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        # Set Passed Parameters
        self.n_agents_per_environment = n_agents_per_environment
        self.n_landmarks_per_environment = n_landmarks_per_environment
        self.n_entities_per_environment = self.n_agents_per_environment + self.n_landmarks_per_environment
        self.n_environments = n_environments

        # For now all agents are size .1, all landmarks are size .5
        self.entity_sizes = torch.zeros((self.n_environments, self.n_entities_per_environment, 1), device=self.device, dtype=torch.float32)
        self.entity_sizes[:] = 0.5 # set all to landmark and then overwrite agent positions
        self.entity_sizes[:, :self.n_agents_per_environment, :] = 0.1 # set agent sizes
        


        # Generic world parameters
        self.dt: float = 0.5 # time step size
        self.damping: float = 0.015
        self.max_speed: float = 20.0
        self.dim_p: int = 2 # agents move x,y
        # Some physics parameters, not sure if cpu or gpu
        self.contact_force: float = 1e3
        self.contact_margin: float = 1e-3


        # Entities have: position, velocity, ctrl parameters
        # TODO: Possibly don't give landmarks velocities or control parameters if possible
        self.positions = torch.zeros((self.n_environments, self.n_entities_per_environment, self.dim_p), device=self.device, dtype=torch.float32)
        self.velocities = torch.zeros((self.n_environments, self.n_entities_per_environment, self.dim_p), device=self.device, dtype=torch.float32)

        # Control parameters
        self.ctrl_thetas = torch.zeros((self.n_environments, self.n_entities_per_environment, 1), device=self.device, dtype=torch.float32)
        self.ctrl_speeds = torch.zeros((self.n_environments, self.n_entities_per_environment, 1), device=self.device, dtype=torch.float32)

        # Agents specifically also have actions, TODO: stop allocating actiosn for entities
        self.agent_actions = torch.zeros((self.n_environments, self.n_entities_per_environment, self.dim_p), device=self.device, dtype=torch.float32)
        self.movables = torch.zeros((self.n_environments, self.n_entities_per_environment, 1), device=self.device, dtype=torch.float32)
        self.movables[:, :self.n_agents_per_environment] = 1 # only agents can move

        # Size matrix for each environment
        self.size_matrix = self.entity_sizes * self.entity_sizes.reshape(self.n_environments, 1, self.n_entities_per_environment)
        
        # This is faster but I'll deal with this later
        #self.heading = torch.cat([torch.sin(self.ctrl_thetas), torch.cos(self.ctrl_thetas)], dim=2) # It is noticably better to update this 

    def mem(self, msg=""):
        print(f"{msg}: {torch.cuda.memory_allocated(self.device)} ")

    def step(self) -> None:
        # Step 1: Update velocities with respect to ctrl_theta and ctrl_speed (modified by RL actions)
        heading: torch.Tensor = torch.cat([torch.sin(self.ctrl_thetas), torch.cos(self.ctrl_thetas)], dim=2)
        
        self.velocities = self.max_speed * (self.ctrl_speeds * self.movables) * heading

        # Step 2: Check collisions
        dist_matrix = torch.cdist(self.positions, self.positions, p=2)

        collisions = dist_matrix < self.size_matrix

        # Step 3: calculate collision forces
        penetrations: torch.Tensor = torch.log(1 + -(dist_matrix - self.size_matrix) / self.contact_margin) * self.contact_margin * collisions
        forces_s: torch.Tensor = self.contact_force * penetrations * collisions
        diff_matrix: torch.Tensor = self.positions[:, :, None, :] - self.positions[:, None, :, :] # broadcasting is so cool
        forces_v = diff_matrix * forces_s[..., None]

        # Step 4: Integrate collision forces
        self.velocities += forces_v.sum(dim=1) * self.dt
        self.velocities[:, self.n_agents_per_environment:, :] = 0 # landmarks really shouldn't be getting actions

        # Step 5: Handle damping... not going to damp for now
        # self.velocities -= self.velocities * (1 - self.damping)

        # Step 6: Integrate position
        self.positions += self.velocities * self.dt
        

    @property
    def landmarks(self) -> torch.Tensor:
        # (n_envs, n_landmarks_per_env, 2)
        return self.positions[:, self.n_agents_per_environment:, :]
    
    @property
    def agents(self) -> torch.Tensor:
        # (n_envs, n_agents_per_env, 2)
        return self.positions[:, :self.n_agents_per_environment, :]

    @property
    def agent_velocities(self) -> torch.Tensor:
        return self.velocities[:, :self.n_agents_per_environment, :]

    def observation(self):
        # (n_envs, n_agents, n_landmarks, 2)
        vec_agents_to_targets = self.landmarks[:, None, :, :] - self.agents[:, :, None, :]
        dist_agents_to_targets = torch.norm(vec_agents_to_targets, dim = 2) # (n_envs, n_agents, n_landmarks)
        # This should be checked just to make sure
        angles_to_targets = torch.atan2(vec_agents_to_targets[:, :, :, 1], vec_agents_to_targets[:, :, :, 0])

        vec_agents_to_agents =  self.agents[:, None, :, :] - self.agents[:, :, None, :] # yes includes self
        dist_agents_to_agents = torch.norm(vec_agents_to_agents, dim = 2)

        vel_agents_to_agents = self.agent_velocities[:, None, :, :] - self.agent_velocities[:, :, None, :]
        speed_agents_to_agents = torch.norm(vel_agents_to_agents, dim = 2)

        # Ryan ordered here but this seems, weird to me, the order should stay the order...
        #breakpoint()
        # (n_envs, n_agents, ?) The last is dependent on the observations but first two required
        obs = torch.cat([dist_agents_to_targets, angles_to_targets, dist_agents_to_agents, speed_agents_to_agents], dim=2)

        return obs

    def reset(self):
        """
        Resets ALL environments
        """
        self.positions = torch.randn(self.positions.shape, device=self.device) # uh idk -3, 3 or something who knows
        self.velocities[:] = 0
        self.agent_actions[:] = 0

        self.ctrl_speeds[:] = 0
        self.ctrl_thetas[:] = 0


    def global_reward(self):
        headings: torch.Tensor = torch.cat([self.ctrl_thetas, self.ctrl_speeds], dim = 2)
        headpos = self.positions + headings * (self.entity_sizes * self.movables)
        
        dists = torch.cdist(headpos, headpos)
        # mask away landmarks from reward
        mask = torch.logical_xor(self.movables, torch.logical_not(self.movables))
        relevant_dists = dists*mask
        # [:, None] to keep shape (n_environments, 1) rather than (n_environments) 
        dist_penalty = relevant_dists.sum(dim=[1,2]).square()[:, None]
        
        diff_matrix = self.positions[:, None, :, :] - self.positions[:, :, None, :]
        rel_thetas = torch.atan2(diff_matrix[:, :, :, 1], diff_matrix[:, :, :, 0])
        coverage_reward = torch.var(rel_thetas)

        return -dist_penalty / (coverage_reward + 1e6)





"""
I just need this one for the time being to more easily build the model lol
"""

class RotatorWorld:
    def __init__(self, n_agents: int, n_landmarks: int, use_gpu: bool=True):
        # Set device
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        # Set passed arguments to object fields
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.n_entities = n_agents + n_landmarks
        self.entity_sizes = [.1 for _ in range(self.n_agents)] + [.5 for _ in range(self.n_landmarks)]
        self.entity_sizes = torch.cuda.FloatTensor(self.entity_sizes)[:,  None]
        assert self.entity_sizes.shape == (self.n_entities, 1), "Entity sizes must be (n_entities, 1) tensor"

        # Generic world parameters
        self.dt: float = 0.5 # time step size
        self.damping: float = 0.015
        self.max_speed: float = 20.0

        # TODO: Figure out if faster to have thse gpu allocated or cpu allocated, think cpu makes sense
        self.dim_p: int = 2 # This can be cpu?
        # Some physics parameters, not sure if cpu or gpu
        self.contact_force: float = 1e3
        self.contact_margin: float = 1e-3

        # Agents and landmarks have positions, only agents have velocity
        self.positions: torch.Tensor = torch.zeros((self.n_entities, self.dim_p), device=self.device, dtype=torch.float32)
        self.velocities: torch.Tensor = torch.zeros((self.n_agents, self.dim_p), device=self.device, dtype=torch.float32)

        # These parameters are mutated by agent actions
        self.ctrl_thetas: torch.Tensor = torch.zeros((self.n_agents, 1), device=self.device, dtype=torch.float32)
        self.ctrl_speeds: torch.Tensor = torch.zeros((self.n_agents, 1), device=self.device, dtype=torch.float32)

        # I think we have to store agent actions?
        #self.agent_actions: torch.Tensor = torch.zeros((self.n_entities, self.dim_p), device=self.device, dtype=torch.float32)        

        # This functions as a bool array but I think CUDA is happier with same type
        self.movables: torch.Tensor = torch.zeros((self.n_entities, 1), device=self.device, dtype=torch.float32)
        self.movables[:self.n_agents] = 1

        # (n_entity, n_entity)
        self.size_matrix: torch.Tensor = self.entity_sizes * self.entity_sizes.T
        self.inv_eye = torch.logical_not(torch.eye(self.n_entities))
        self.inv_eye = self.inv_eye.to(self.device)
        #dist_matrix = torch.cdist(self.positions, self.positions, p=2)
        self.mem("init")
        self.batch_size = 10
        self.n_batches = 100

        self.agent_rewards: torch.Tensor = torch.zeros(n_agents)

    def mem(self, msg=""):
        print(f"{msg}: {torch.cuda.memory_allocated(self.device)/1_048_576:.2f} ")
    

    def step(self) -> None:
        """
        Steps all agents, its pretty inefficient rn, need to do less reshapes because that'll
        do dumb shit in GPU memory
        """
        # No need to Transpose, should be refactored such that this isn't done
        
        heading: torch.Tensor = torch.cat([torch.sin(self.ctrl_thetas), torch.cos(self.ctrl_thetas)], dim=1)
        #breakpoint()
        self.velocities = self.max_speed * self.ctrl_speeds * heading
        #breakpoint()
        # Step 2: Check collisions
        # Prolly should make sure this is the same as distance matrix
        dist_matrix = torch.cdist(self.positions, self.positions, p=2)
        #breakpoint()
        #self.mem("dist_matrix")   
        collisions = dist_matrix < self.size_matrix # TODO: Handle Diagonal bug
        #collisions = self.inv_eye * collisions 
        #breakpoint()
        #self.mem("collisions")     
        # Step 3: calculate collision forces .... the * self.inv_eye is to mask away the diagonal... solves nan problems
        penetrations: torch.Tensor = torch.log(1 + torch.exp(-(dist_matrix - self.size_matrix)*self.inv_eye / self.contact_margin)) * self.contact_margin * collisions
        #breakpoint()
        #self.mem("penetrations") 
        forces_s = torch.float32 = self.contact_force * penetrations * collisions
        #breakpoint()
        # Need better way of calculating this
        #self.mem("forces_s") 
        diff_matrix: torch.Tensor = self.positions[:, None, :] - self.positions
        #breakpoint()
        #self.mem("diff_matrix") 
        forces_v = diff_matrix * forces_s[..., None]
        #breakpoint()
        #self.mem("forces_v") 
        # Step 4: Integrate collision forces
        self.velocities += forces_v.sum(dim=0)[:self.n_agents, :] * self.dt
        #breakpoint()
        
        #self.velocities[self.n_agents:, :] = 0 # landmarks shouldn't have speed

        # Step 5: Handle damping... not going to damp for now
        # self.velocities -= self.velocities * (1 - self.damping) 
        # Step 6: Integrate position
        self.positions[:self.n_agents, :] += self.velocities * self.dt
        
     
    
    #TODO: Figure out if these properties make copies or if they do something smarter...
    @property
    def landmarks(self) -> torch.Tensor:
        return self.positions[self.n_agents:, :]
    
    @property
    def agents(self) -> torch.Tensor:
        return self.positions[:self.n_agents, :]
    


    def observation(self):
        """
        Returns tensor containing:
        [dist_to_targets, angles_to_targets, dist_agents, speed_to_agents]_i
        for all agents that is:
            a_i = [dist_to_targets, angles_to_targets, dist_agents, speed_to_agents]_i
            tensors = [a_0, a_1, a_2... a_n_agents]
        """
        #TODO: Make this less inefficient
        # Observations does not have to be "good" right now, just interested in proof of concept
        
        dist_agents_to_target: torch.Tensor = torch.cdist(self.agents, self.landmarks)
        angles_agents_to_target: torch.Tensor = torch.cdist(self.agents, self.landmarks) # Yes the angles are literally nothign rn
        dist_agents_to_agents: torch.Tensor = torch.cdist(self.agents, self.agents)
        vel_agents_to_agents: torch.Tensor = torch.cdist(self.velocities, self.velocities)
        #speed_agents_to_agents: torch.Tensor = vel_agents_to_agents.norm(dim=1)[:, None]) #This needs to be checked
        
        # No need for ordering just hoping the sizing makes sense
        # also yes this includes self, whatever its dumb
        #breakpoint()
        obs: torch.Tensor = torch.cat([dist_agents_to_target, angles_agents_to_target, dist_agents_to_agents, vel_agents_to_agents], dim=1)
        #breakpoint()
        return obs
    
    # No need for random seed
    def reset(self) -> None:
        # Prolly should put range of numbers here
        self.positions = torch.randn(self.positions.shape, device=self.device)
        self.velocities[:] = 0
        #self.agent_actions[:] = 0

        self.ctrl_speeds[:] = 0
        self.ctrl_thetas[:] = 0
    
    # I actually want reward to be defined by world type
    # also this implementation is just a tad stupid but whatever
    def global_reward(self):
        headings: torch.Tensor = torch.cat([self.ctrl_thetas, self.ctrl_speeds], dim=1)
        headpos = self.positions[:, :]
        #breakpoint()
        headpos[:self.n_agents, :] += headings * (self.entity_sizes[:self.n_agents, :])

        dists = torch.cdist(headpos, headpos)

        mask = torch.logical_xor(self.movables, torch.logical_not(self.movables.T))
        relevant_dists = dists*mask
        dist_penalty = relevant_dists.sum().square()

        diff_mat = self.positions[:, None, :] - self.positions
        rel_thetas = torch.atan2(diff_mat[:, :, 1], diff_mat[:, :, 0])
        coverage_reward = torch.var(rel_thetas)
        self.agent_rewards[:] = -dist_penalty / (coverage_reward +  1e-6) # convenient way of setting memory
        return -dist_penalty / (coverage_reward +  1e-6)


from utils import Box
import numpy as np # Should eventually switch to not numpy?
class MultiRotatorEnvironment:
    def __init__(self, n_agents: int, n_landmarks: int, n_worlds: int):
        self.device = torch.device("cuda")
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.n_worlds = n_worlds
        self.world = MultiEnvRotatorWorld(self.n_agents, self.n_landmarks, self.n_worlds)
        self.max_cycles = 25 # Just a random default all our agents "finish" at the same b/c we're not retarded
        self.current_step_mod_max_cycles = 0 # we can use this to control when done... swarmcover updated all dones every max_cycles steps
        
        self.action_dim = 2
        self.obs_dim = 7 # apparently 7 lolol

        self._setup_observation_and_action_space()

    def _setup_observation_and_action_space(self):
        obs_min = torch.cat([ 
                    torch.tensor([-np.inf] * self.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([-np.pi] * self.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([-np.inf] * self.n_agents, device=self.device, dtype=torch.float32),
                    torch.tensor([0] * self.n_agents, device=self.device, dtype=torch.float32),
        ])



        obs_max = torch.cat([
                    torch.tensor([np.inf] * self.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([np.pi] * self.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([np.inf] * self.n_agents, device=self.device, dtype=torch.float32),
                    torch.tensor([self.world.max_speed * 2] * self.n_agents, device=self.device, dtype=torch.float32),
        ])

        self.observation_space = Box(low=obs_min,
                                            high=obs_max,
                                            dtype=torch.float32)
        
        # Action space
        self.action_space = Box(low=torch.tensor([-torch.pi, 0], device=self.device, dtype=torch.float32),
                                       high=torch.tensor([torch.pi, 1], device=self.device, dtype=torch.float32),
                                       dtype=np.float32)


    """
    just returns observations and rewards after stepping the world
    """
    def step(self, actions):
        # I'm not copying atm... this might cause really bad bugs we will see
        #breakpoint()
        self.world.ctrl_thetas = actions[:, :, 0, None] # the None is to keep the dimension nice
        self.world.ctrl_speeds = actions[:, :, 1, None]
        #breakpoint()
        self.world.step()
        self.current_step_mod_max_cycles += 1
        self.current_step_mod_max_cycles %= self.max_cycles
        return self.world.observation(), self.world.global_reward(), self.current_step_mod_max_cycles == 0

    def reset(self):
        self.world.reset()
        return self.world.observation()


"""
Having environment contain the world, observation space, action space and whatever
else i need to throw in it... basically just exists so I don't pollute existing classes atm
Eventually this should be deleted
"""
from utils import Box
import numpy as np # Should eventually switch to not numpy?
class RotatorEnvironment:
    def __init__(self):
        self.world = RotatorWorld(5,1, True)
        self.device = torch.device("cuda")
        self.n_agents = self.world.n_agents
        self.n_observations_per_agent = 12 # do this but smarter later
        self.n_action_parameters_per_agent = 2

        self._setup_observation_and_action_space()
        
        self.max_cycles = 25 # This controls dones in stablebaselines implementation

        self.current_step_mod_max_cycles = 0 # we can use this to control when done... swarmcover updated all dones every max_cycles steps


    def _setup_observation_and_action_space(self):
        # Observation space
        # torch.tensor([-np.inf] * self.world.n_landmarks, device=self.device, dtype=torch.float32)
        '''
        obsmin = torch.cat([[-np.inf] * self.world.n_landmarks,
                                [-np.pi] * self.world.n_landmarks, 
                                [-np.inf] * (self.world.n_agents ),
                                [0] * (self.world.n_agents )], device=self.device)
        obsmax = torch.cat([[np.inf] * self.world.n_landmarks,
                                [np.pi] * self.world.n_landmarks, 
                                [np.inf] * (self.world.n_agents ),
                                [self.world.max_speed * 2] * (self.world.n_agents)])
        '''
        


        obs_min = torch.cat([ 
                    torch.tensor([-np.inf] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([-np.pi] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([-np.inf] * self.world.n_agents, device=self.device, dtype=torch.float32),
                    torch.tensor([0] * self.world.n_agents, device=self.device, dtype=torch.float32),
        ])



        obs_max = torch.cat([
                    torch.tensor([np.inf] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([np.pi] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([np.inf] * self.world.n_agents, device=self.device, dtype=torch.float32),
                    torch.tensor([self.world.max_speed * 2] * self.world.n_agents, device=self.device, dtype=torch.float32),
        ])

        self.observation_space = Box(low=obs_min,
                                            high=obs_max,
                                            dtype=torch.float32)
        
        # Action space
        self.action_space = Box(low=torch.tensor([-torch.pi, 0], device=self.device, dtype=torch.float32),
                                       high=torch.tensor([torch.pi, 1], device=self.device, dtype=torch.float32),
                                       dtype=np.float32)
    

    """
    just returns observations and rewards after stepping the world
    """
    def step(self, actions):
        # I'm not copying atm... this might cause really bad bugs we will see
        #breakpoint()
        self.world.ctrl_thetas = actions[:, 0, None] # the None is to keep the dimension nice
        self.world.ctrl_speeds = actions[:, 1, None]
        #breakpoint()
        self.world.step()
        self.current_step_mod_max_cycles += 1
        self.current_step_mod_max_cycles %= self.max_cycles
        return self.world.observation(), self.world.global_reward(), self.current_step_mod_max_cycles == 0

    def reset(self):
        self.world.reset()
        return self.world.observation()




if __name__ == "__main__":
    env = MultiRotatorEnvironment()
    print(env.reset().shape)
    #world.observation()
    #print(world.global_reward().shape)


