#include <c10/core/Device.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAMathCompat.h>


void dummy();

class RotatorWorld {
    private:
        c10::Device _device = c10::Device("cpu");
        // has to be int64 because torch expects 64 bit
        int64_t _num_landmarks, _num_agents, _num_entities, _dim_p; 
        float _dt, _damping, _max_speed, _contact_force, _contact_margin;

        torch::Tensor _positions, _velocities, _ctrl_thetas, _ctrl_speeds, _size_matrix, _entity_sizes, _inv_eye, _tmp;
    public:
        explicit RotatorWorld(int, int, bool);
        void step();
        void observation();
        void reset();
        void reward();
};