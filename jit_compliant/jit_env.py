import torch
from jit_utils import Box
import numpy as np



@torch.jit.script
class RotatorWorldJit:
    def __init__(self, n_agents: int, n_landmarks: int, use_gpu: bool=True):
        # Set device
        self.device: torch.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        # Set passed arguments to object fields
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.n_entities = n_agents + n_landmarks
        entity_sizes_list = [.1 for _ in range(self.n_agents)] + [.5 for _ in range(self.n_landmarks)]
        self.entity_sizes: torch.Tensor = torch.tensor(entity_sizes_list, device=self.device, dtype=torch.float32)[:,  None]
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
       

        #self.agent_rewards: torch.Tensor = torch.zeros(n_agents)

        # we do not want to recompute at every call to global_reward
        self.movables_mask: torch.Tensor = torch.logical_xor(self.movables, torch.logical_not(self.movables.T))
        self.tmp = torch.zeros((10,10))
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
        dist_matrix = torch.cdist(self.positions, self.positions, p=2.0)
        #breakpoint()
        #self.mem("dist_matrix")   
        
        #collisions = dist_matrix < self.size_matrix # TODO: Handle Diagonal bug
        #collisions = self.inv_eye * collisions 
        
        
        #breakpoint()
        #self.mem("collisions")     
        # Step 3: calculate collision forces .... the * self.inv_eye is to mask away the diagonal... solves nan problems
        #penetrations: torch.Tensor = torch.log_(1 + torch.exp(-(dist_matrix - self.size_matrix)*self.inv_eye / self.contact_margin)) * self.contact_margin * collisions
        #penetrations = -(dist_matrix - self.size_matrix)/self.contact_margin
        self.tmp = dist_matrix
        #print(self.tmp)
        #breakpoint()
        #self.mem("penetrations") 
        #forces_s: torch.Tensor = self.contact_force * penetrations * collisions
        #breakpoint()
        # Need better way of calculating this
        #self.mem("forces_s") 
        #diff_matrix: torch.Tensor = self.positions[:, None, :] - self.positions
        #breakpoint()
        #self.mem("diff_matrix") 
        #forces_v = diff_matrix * forces_s[..., None]
        #breakpoint()
        #self.mem("forces_v") 
        # Step 4: Integrate collision forces
        #self.velocities += forces_v.sum(dim=0)[:self.n_agents, :] * self.dt
        #breakpoint()
        
        #self.velocities[self.n_agents:, :] = 0 # landmarks shouldn't have speed

        # Step 5: Handle damping... not going to damp for now
        # self.velocities -= self.velocities * (1 - self.damping) 
        # Step 6: Integrate position
        #self.positions[:self.n_agents, :] += self.velocities * self.dt
        
     
    
    #TODO: Figure out if these properties make copies or if they do something smarter...
    #@torch.jit.export
    @property
    def landmarks(self) -> torch.Tensor:
        return self.positions[self.n_agents:, :]
    
    #@torch.jit.export
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

        
        relevant_dists = dists*self.movables_mask
        dist_penalty = relevant_dists.sum().square()

        diff_mat = self.positions[:, None, :] - self.positions
        rel_thetas = torch.atan2(diff_mat[:, :, 1], diff_mat[:, :, 0])
        coverage_reward = torch.var(rel_thetas)
        #self.agent_rewards[:] = -dist_penalty / (coverage_reward +  1e-6) # convenient way of setting memory... this is sorta stupid since all rewards are the same rn
        return -dist_penalty / (coverage_reward +  1e-6)
    




@torch.jit.script
class RotatorEnvironmentJit:
    def __init__(self):
        self.world: RotatorWorldJit = RotatorWorldJit(5,1, True)
        #print(self.world)
        self.device: torch.device = torch.device("cuda")
        self.n_agents: int = self.world.n_agents
        self.n_observations_per_agent: int = 12 # do this but smarter later
        self.n_action_parameters_per_agent: int = 2
        

        # Yea so like torch is mean and I have to do this here
        obs_min = torch.cat([ 
                    torch.tensor([-np.inf] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([-np.pi] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([-np.inf] * self.world.n_agents, device=self.device, dtype=torch.float32),
                    torch.tensor([0.0] * self.world.n_agents, device=self.device, dtype=torch.float32),
        ])



        obs_max = torch.cat([
                    torch.tensor([np.inf] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([np.pi] * self.world.n_landmarks, device=self.device, dtype=torch.float32),
                    torch.tensor([np.inf] * self.world.n_agents, device=self.device, dtype=torch.float32),
                    torch.tensor([self.world.max_speed * 2.0] * self.world.n_agents, device=self.device, dtype=torch.float32),
        ])

        self.observation_space = Box(low=obs_min,
                                            high=obs_max,
                                            dtype=torch.float32)
        
        # Action space
        self.action_space = Box(low=torch.tensor([-torch.pi, 0.0], device=self.device, dtype=torch.float32),
                                       high=torch.tensor([torch.pi, 1.0], device=self.device, dtype=torch.float32),
                                       dtype=torch.float32)
        #self._setup_observation_and_action_space()
        
        self.max_cycles: int = int(25) # This controls dones in stablebaselines implementation (RotatorCoverage has this at 100, the extent this matters for our use case is tiny... prolly)

        self.current_step_mod_max_cycles: int = int(0) # we can use this to control when done... swarmcover updated all dones every max_cycles steps




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
        self.current_step_mod_max_cycles = int(self.current_step_mod_max_cycles % self.max_cycles) # what the fuck torch... can't %=
        return self.world.observation(), self.world.global_reward(), self.current_step_mod_max_cycles == 0
    
    def reset(self):
        self.world.reset()
        return self.world.observation()





if __name__ == "__main__":
    #box = Box()
    
    #env = torch.jit.script(RotatorEnvironmentJit())
    #a = env.observation_space
    world = torch.jit.script(RotatorWorldJit(500,1,True))
    #breakpoint()
    import time
    print("Staring...")
    s = time.time()
    for _ in range(20_000):
        #world.step()
        world.step()
        torch.cuda.synchronize()
        #env.reset()
    print(f"Time: {time.time() - s}")