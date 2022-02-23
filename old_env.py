import torch
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

        # Agents have a position and a velocity
        self.positions: torch.Tensor = torch.zeros((self.n_entities, self.dim_p), device=self.device, dtype=torch.float32)
        self.velocities: torch.Tensor = torch.zeros((self.n_entities, self.dim_p), device=self.device, dtype=torch.float32)

        # These parameters are mutated by agent actions
        self.ctrl_thetas: torch.Tensor = torch.zeros((self.n_entities, 1), device=self.device, dtype=torch.float32)
        self.ctrl_speeds: torch.Tensor = torch.zeros((self.n_entities, 1), device=self.device, dtype=torch.float32)

        # I think we have to store agent actions?
        self.agent_actions: torch.Tensor = torch.zeros((self.n_entities, self.dim_p), device=self.device, dtype=torch.float32)        

        # This functions as a bool array but I think CUDA is happier with same type
        self.movables: torch.Tensor = torch.zeros((self.n_entities, 1), device=self.device, dtype=torch.float32)
        self.movables[:self.n_agents] = 1

        # (n_entity, n_entity)
        self.size_matrix = self.entity_sizes * self.entity_sizes.T
        self.inv_eye = torch.logical_not(torch.eye(self.n_entities))
        self.inv_eye = self.inv_eye.to(self.device)
        #dist_matrix = torch.cdist(self.positions, self.positions, p=2)
        self.mem("init")
        self.batch_size = 10
        self.n_batches = 100        
    def mem(self, msg=""):
        print(f"{msg}: {torch.cuda.memory_allocated(self.device)/1_048_576:.2f} ")
    

    def step(self) -> None:
        """
        Steps all agents, its pretty inefficient rn, need to do less reshapes because that'll
        do dumb shit in GPU memory
        """
        # No need to Transpose, should be refactored such that this isn't done
        
        heading: torch.Tensor = torch.cat([torch.sin(self.ctrl_thetas), torch.cos(self.ctrl_thetas)], dim=1)

        self.velocities = self.max_speed * (self.ctrl_speeds * self.movables) * heading
        
        # Step 2: Check collisions
        # Prolly should make sure this is the same as distance matrix
        dist_matrix = torch.cdist(self.positions, self.positions, p=2)
        #self.mem("dist_matrix")   
        collisions = dist_matrix < self.size_matrix # TODO: Handle Diagonal bug
        #self.mem("collisions")     
        # Step 3: calculate collision forces
        penetrations: torch.Tensor = torch.log(1 + -(dist_matrix - self.size_matrix) / self.contact_margin) * self.contact_margin * collisions
        #self.mem("penetrations") 
        forces_s = torch.float32 = self.contact_force * penetrations * collisions
        # Need better way of calculating this
        #self.mem("forces_s") 
        diff_matrix: torch.Tensor = self.positions[:, None, :] - self.positions
        #self.mem("diff_matrix") 
        forces_v = diff_matrix * forces_s[..., None]
        #self.mem("forces_v") 
        # Step 4: Integrate collision forces
        self.velocities += forces_v.sum(dim=0) * self.dt
        
        self.velocities[self.n_agents:, :] = 0 # landmarks shouldn't have speed

        # Step 5: Handle damping... not going to damp for now
        # self.velocities -= self.velocities * (1 - self.damping) 
        # Step 6: Integrate position
        self.positions += self.velocities * self.dt
        
     
    
    #TODO: Figure out if these properties make copies or if they do something smarter...
    @property
    def landmarks(self) -> torch.Tensor:
        return self.positions[self.n_agents:, :]
    
    @property
    def agents(self) -> torch.Tensor:
        return self.positions[:self.n_agents, :]
    
    @property
    def agent_velocities(self) -> torch.Tensor:
        return self.velocities[:self.n_agents, :]

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
        vel_agents_to_agents: torch.Tensor = torch.cdist(self.agent_velocities, self.agent_velocities)
        #speed_agents_to_agents: torch.Tensor = vel_agents_to_agents.norm(dim=1)[:, None]) #This needs to be checked

        # No need for ordering just hoping the sizing makes sense
        # also yes this includes self, whatever its dumb
        obs: torch.Tensor = torch.cat([dist_agents_to_target, angles_agents_to_target, dist_agents_to_agents, vel_agents_to_agents], dim=1)

        return obs
    
    # No need for random seed
    def reset(self) -> None:
        # Prolly should put range of numbers here
        self.positions = torch.randn(self.positions.shape)
        self.velocities[:] = 0
        self.agent_actions[:] = 0

        self.ctrl_speeds[:] = 0
        self.ctrl_thetas[:] = 0
    
    # I actually want reward to be defined by world type
    # also this implementation is just a tad stupid but whatever
    def global_reward(self):
        headings: torch.Tensor = torch.cat(self.ctrl_thetas, self.ctrl_speeds)
        headpos = self.positions + headings * (self.entity_sizes * self.movables)

        dists = torch.cdist(headpos, headpos)

        mask = torch.logical_xor(self.movables, ~self.movables)
        relevant_dists = dists*mask
        dist_penalty = relevant_dists.sum().square()

        diff_mat = self.positions[:, None, :] - self.positions
        rel_thetas = torch.arctan(diff_mat[:, :, 1], diff_mat[:, :, 0])
        coverage_reward = torch.var(rel_thetas)

        return -dist_penalty / (coverage_reward +  1e-6)

import numpy as np
from scipy.spatial import distance_matrix

class FastWorld:
    
    def __init__(self, n_agents: int, n_entities: int, entity_sizes: np.ndarray):
        self.n_agents = n_agents
        self.n_landmarks = n_entities - n_agents
        self.n_entities = n_entities
        self.entity_sizes = entity_sizes

        # World parameters
        self.dt = 0.5
        self.damping = 0.015
        self.maxspeed = 20.0

        self.dim_p = 2 # (x,y)
        self.contact_force = np.float32(1e3)
        self.contact_margin = np.float32(1e-3)


        # Agent controls, every agent has an X, Y positon
        self.positions: np.ndarray = np.zeros(shape=(n_entities, self.dim_p))
        self.velocities: np.ndarray = np.zeros(shape=(n_entities, self.dim_p))

        self.ctrl_thetas: np.ndarray = np.zeros(n_entities)
        self.ctrl_speeds: np.ndarray = np.zeros(n_entities)#[:, None]

        # Agent can do an action, action space for each agent is (X, Y)
        self.agent_actions: np.ndarray[np.ndarray] = np.zeros(shape=(n_entities, self.dim_p))

        # Only agents are movable
        self.movables = np.zeros(n_entities, dtype=bool)
        #self.movables[:] = False
        self.movables[:n_agents] = 1
        self.targets = self.movables[:]
        #breakpoint()
        self.targets = np.ones(n_entities,dtype=bool) #[:, None]
        self.targets[:n_agents] = 0

        self.entity_sizes: np.ndarray = entity_sizes

        self.sizemat = self.entity_sizes[..., None] * self.entity_sizes[None, ...]
        #self.sizemat *= 2 # this is a guess... radius vs complete circle

        self.diag_indices = np.diag_indices(self.positions.shape[0])
        

    def step(self) -> None:
        # heading.shape = 2,6,2
        heading: np.ndarray = np.vstack([np.cos(self.ctrl_thetas), np.sin(self.ctrl_thetas)])
        heading = heading.T
        #print(f"heading: {heading.shape}, velocity: {self.velocities.shape} ")
        #print(f"ctrl_speed: {self.ctrl_speeds.shape} movables: {self.movables.shape}")
        self.velocities = self.maxspeed * (self.ctrl_speeds * self.movables)[:, None] * heading
        
        #self.velocities = self.ctrl_speeds * heading * self.targets
        #assert self.velocities.shape == (6,2), "self.velocities shape mutated"

        # 2 detect collisons
        dist_matrix = distance_matrix(self.positions, self.positions)
        self.collisions: np.ndarray = dist_matrix < self.sizemat
        self.collisions[self.diag_indices] = False
        #self.collisions[self.low_triangular_indices_positions] = False
        # prolly a smarter way to check
        if np.any(self.collisions):
            #print(f"Collisons detected: {collisions}")
            # 3 calculate collison forces  
            penetrations = np.logaddexp(0, -(dist_matrix - self.sizemat) / self.contact_margin) \
                                * self.contact_margin * self.collisions
            

            forces_s: np.float32 = self.contact_force * penetrations * self.collisions
            diffmat: np.ndarray = self.positions[:, None, :] - self.positions  # skew symetric
            
            forces_v = diffmat * forces_s[..., None]
            
            #breakpoint()

            # 4 integrate collsion forces
            s = np.sum(forces_v, axis=0)
            #print(f"np.sum(forces_v, axis=0).shape {s.T.shape}")
            #print(f"self.velocities.shape {self.velocities.shape}")
            #print(f"self.dt {self.dt}")
            self.velocities += np.sum(forces_v, axis=0) * self.dt
            self.velocities[self.n_agents:, :] = 0
            #assert np.all(self.velocities[self.n_agents:]) == 0, "Sanity check, landmarks don't move" 
        # 5 integrate damping
        self.velocities -= self.velocities * (1 - self.damping)
        #print(self.velocities)
        assert np.all(self.velocities[self.n_agents:]) == 0, "Sanity check, landmarks don't move" 
        # Integrate position
        self.positions += self.velocities * self.dt



    @property
    def landmarks(self) -> np.ndarray:
        return self.positions[self.n_agents:, :]

    @property
    def agents(self) -> np.ndarray:
        return self.positions[:self.n_agents, :]

    @property
    def agent_velocities(self) -> np.ndarray:
        return self.velocities[:self.n_agents, :]

    def observation(self, agent_index) -> np.ndarray:
        """
        WARNING: DOES NOT RETURN COMMUNICATION
        """
        # Calculate distances and velocities
        vec_to_targets = self.landmarks - self.agents[agent_index]  # targets
        dist_to_targets = np.linalg.norm(vec_to_targets, axis=1)
        vec_to_agents = self.agents - self.agents[agent_index]      # agents
        dist_to_agents = np.linalg.norm(vec_to_agents, axis=1)
        vel_to_agents = self.agent_velocities[agent_index] - self.agent_velocities
        speed_to_agents = np.linalg.norm(vel_to_agents, axis=1)

        # Calculate angles to landmarks
        angles_to_targets = np.arctan2(vec_to_targets[:, 1], vec_to_targets[:, 0])

        # Agents / landmarks have no intrinsic order, so we sort by distance
        targets_order = np.argsort(dist_to_targets)             # targets
        dist_to_targets = dist_to_targets[targets_order]
        angles_to_targets = angles_to_targets[targets_order]
        agents_order = np.argsort(dist_to_agents)               # agents
        dist_to_agents = dist_to_agents[agents_order]
        speed_to_agents = speed_to_agents[agents_order]

        obs = np.concatenate([dist_to_targets, angles_to_targets, dist_to_agents[1:], speed_to_agents[1:]])


        # Don't forget that we included the agent itself in the list of others
        return obs


    def reset(self, np_random) -> None:
        """
        Resets the world
        """
        self.positions =  np_random.uniform(-3, +3, size=(self.positions.shape))
        self.velocities[:] = 0
        self.agent_actions[:] = 0

        self.ctrl_thetas[:] = 0
        self.ctrl_speeds[:] = 0



'''
world = RotatorWorld(2_999, 1)
for _ in range(50_000):
    world.step()

n_agents = 599
entities = [.1 for _ in range(n_agents)] + [.5]
entities = np.array(entities, dtype=np.float32)
npw = FastWorld(n_agents, n_agents+1, entities)
s = time.time()
for _ in range(100):
    npw.step()

print(s-time.time())

world = RotatorWorld(n_agents, 1, use_gpu=True)


#world.step()

n_runs = 5
runs = []

for _ in range(n_runs):
    s = time.time()
    for _ in range(10_000):
        world.step()
    runs.append(time.time()-s)
print(f"time: {np.mean(runs)}")

'''
