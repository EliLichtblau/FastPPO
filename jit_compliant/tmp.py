import torch
import numpy as np
@torch.jit.script
class Box:
    def __init__(self, low: torch.Tensor, high: torch.Tensor, dtype: torch.dtype):
        self.low = low
        self.high = high
        self.dtype = dtype
        assert low.shape == high.shape, "Box low and high shapes must be the same"
        self.shape = low.shape


device = torch.device("cuda")
obs_min = torch.cat([ 
            torch.tensor([-np.inf] * 1, device=device, dtype=torch.float32),
            torch.tensor([-np.pi] * 1, device=device, dtype=torch.float32),
            torch.tensor([-np.inf] * 5, device=device, dtype=torch.float32),
            torch.tensor([0] * 5, device=device, dtype=torch.float32),
])



obs_max = torch.cat([
            torch.tensor([np.inf] * 1, device=device, dtype=torch.float32),
            torch.tensor([np.pi] * 1, device=device, dtype=torch.float32),
            torch.tensor([np.inf] * 5, device=device, dtype=torch.float32),
            torch.tensor([20.0 * 2] * 5, device=device, dtype=torch.float32),
])


b = Box(obs_min, obs_max, torch.float32)

@torch.jit.script
class RotatorWorldJit:
    def __init__(self, n_agents: int, n_landmarks: int, use_gpu: bool=True):
        # Set device
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
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
        self.batch_size = 10
        self.n_batches = 100

        self.agent_rewards: torch.Tensor = torch.zeros(n_agents)