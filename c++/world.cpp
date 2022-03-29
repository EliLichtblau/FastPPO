#include <torch/torch.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>


#include <c10/cuda/CUDAMathCompat.h>

#include <iostream>

#include "world.h"

void dummy() {
    std::cout << "dummy" << std::endl;
}



RotatorWorld::RotatorWorld(int n_agents, int n_landmarks, bool use_gpu) {
    c10::DeviceType kDevice = torch::kCUDA;
    if (use_gpu) {
        _device = c10::Device("cuda");
    } else {
        _device = c10::Device("cpu");
        kDevice = torch::kCPU;
    }
    
    // set the number of agents, landmarks in environment
    _num_agents = n_agents;
    _num_landmarks = n_landmarks;
    _num_entities = n_agents + n_landmarks;
    
    // set world parameters
    _dim_p = 2;
    _dt = 0.5;
    _damping = 0.015;
    _max_speed = 20.0;
    _contact_force = 1e3;
    _contact_margin = 1e3;


    // options for everything
    auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(kDevice)
                    .requires_grad(false)
                    ;

    {
        using namespace torch::indexing;
        //_entity_sizes = torch::zeros({_num_entities, 1}, options=options);
        _entity_sizes = torch::full({_num_entities, 1}, 0.15, options=options);
        _entity_sizes.index_put_({Slice(None, 3)}, 1);
        
    }
    
    _positions = torch::zeros({_num_entities, _dim_p}, options=options);
    _velocities = torch::zeros({_num_agents, _dim_p}, options=options);

    _ctrl_thetas = torch::zeros({_num_agents, 1}, options=options);
    _ctrl_speeds  = torch::zeros({_num_agents, 1}, options=options);
    
    std::cout << "here" << std::endl;
    // (n_entites, n_entites)
    _size_matrix = _entity_sizes * _entity_sizes.transpose(0,1);
    //std::cout << _entity_sizes << std::endl;
    //std::cout << _entity_sizes.transpose(0,1) << std::endl;
    //std::cout << _entity_sizes.sizes() << std::endl;
    _inv_eye = torch::eye(_num_entities, options=options).logical_not();
    //_inv_eye = torch::logical_not()
}


void RotatorWorld::step() {
    torch::Tensor heading = torch::cat({torch::sin(_ctrl_thetas), torch::cos(_ctrl_thetas)}, 1);

    _velocities = _max_speed * _ctrl_speeds * heading;

    torch::Tensor dist_matrix = torch::cdist(_positions, _positions);
    // make sure to set diagonal to 0
    torch::Tensor collisions =  (dist_matrix < _size_matrix) * _inv_eye; 

    //_tmp = collisions;
    {
        torch::Tensor penetrations = ((-(dist_matrix - _size_matrix) / _contact_margin) );
        _tmp = penetrations; 
    }
    //torch::Tensor penetrations = torch::exp(-(dist_matrix - _size_matrix) / _contact_margin); //* _contact_margin * collisions;
    
    //penetrations = torch::log(1 + penetrations) * _contact_margin * collisions;
    /*torch::Tensor forces_s = _contact_force * penetrations * collisions;
    
    // look into making this less stupid
    //torch::Tensor diff_matrix = _positions.unsqueeze(1) - _positions;

    // possible wrong dimension
    torch::Tensor forces_v = forces_s.unsqueeze(forces_s.sizes().size());
    {
        using namespace torch::indexing;
        _velocities += forces_v.sum(0, false).index({Slice(None, _num_agents), Slice() }) * _dt;
        // _positions[:n_agents, :] += vel * dt
        _positions.index_put_({Slice(None, _num_agents), Slice()}, _positions.index({Slice(None, _num_agents), Slice()}) + _velocities * _dt);
    }*/
       
}

