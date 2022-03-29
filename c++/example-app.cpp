#include <torch/torch.h>
#include <iostream>
#include "world.h"
#include <chrono>

int main() {
     
    std::cout << torch::cuda::is_available() << std::endl;
    
    //torch::Tensor tensor = at::tensor({ -1, 1 }, at::kCUDA);


    //torch::Tensor tensor = torch::randn({3, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(true));
 

//torch::rand({2, 3});
  //std::cout << tensor << std::endl;

  dummy();
  RotatorWorld world = RotatorWorld(5,1, true);

  std::cout << "Starting" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 20'000; i++)
    world.step();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time: " << duration.count() << std::endl;

}