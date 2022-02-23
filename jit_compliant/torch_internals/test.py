import torch
from torch import _VF
from typing import List, Union, Optional, Tuple, Dict, Any

from torch.overrides import has_torch_function
from numbers import Number
@torch.jit.script
def test(a: torch.Tensor, b: torch.Tensor):
    return a * b






@torch.jit.script
def _broadcast_shape_two_tensors(shape_a: List[int], shape_b: List[int]) -> Tuple[bool, List[int]]:
    shape = list(range(len(shape_a)))
    if len(shape_a) != len(shape_b):
        return False, shape

    
    for i, (a,b) in enumerate(zip(shape_a, shape_b)):
        if a != b:
            if a !=1 and b != 1:
                return False, shape
        shape[i] = a
        if a == 1:
            shape[i] = b
    return True, shape



@torch.jit.script
def _broadcast_shape_over_list(tensor_list: List[torch.Tensor]) -> List[int]:
    shape: Optional[List[int]] = None
    for tensor in tensor_list:
        if shape is None:
            shape = list(tensor.shape)
        #print(shape)
        broadcastable, shape = _broadcast_shape_two_tensors(shape, tensor.shape)
        if broadcastable is False:
            raise ValueError("Passed Tensors not broadcastable")
    
    if shape is None:
        raise ValueError("Whatever you passed is garbadge")
    return shape

        
@torch.jit.script
def broadcast_tensors(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    shape = _broadcast_shape_over_list(tensor_list)
    return [torch.broadcast_to(tensor, shape) for tensor in tensor_list]

def broadcast_all(*values):
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all(isinstance(v, torch.Tensor) or has_torch_function((v,)) or isinstance(v, Number)
               for v in values):
        raise ValueError('Input arguments must all be instances of numbers.Number, '
                         'torch.Tensor or objects implementing __torch_function__.')
    if not all([isinstance(v, torch.Tensor) or has_torch_function((v,)) for v in values]):
        options: Dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [v if isinstance(v, torch.Tensor) or has_torch_function((v,)) else torch.tensor(v, **options)
                      for v in values]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)


x = torch.arange(3).view(1, 3, 1)
y = torch.arange(2).view(2, 1, 1)
z = torch.arange(4).view(1, 1, 4)

import time
s = time.time()
for _ in range(100_000):
    a,b,c = broadcast_all(x,y,z)
print(f"Time: {time.time() - s}")



class BroadCast(torch.nn.Module):
    def __init__(self):
        super(BroadCast, self).__init__()
    def forward(self, x: List[torch.Tensor]):
        return _VF.broadcast_tensors(x)

mod = torch.jit.script(BroadCast())

a,b,c = mod([x, y, z])

s = time.time()
l = [x,y,z]
for _ in range(100_000):
    a,b,c = mod.forward(l)
print(f"Time: {time.time() - s}")

