import torch
from torch.overrides import has_torch_function
from numbers import Real
from numbers import Number
from typing import Dict, Any, Optional
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions import constraints
import math

from typing import List


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




class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)









import warnings

'''

@torch.jit.script
class Distribution(object):
    has_rsample: bool = False
    has_enumerate_support: bool = False    
    _valid_args = __debug__

    @staticmethod
    @torch.jit.export
    def set_default_validate_args(value):
        """
        Sets whether validation is enabled or disabled.

        The default behavior mimics Python's ``assert`` statement: validation
        is on by default, but is disabled if Python is run in optimized mode
        (via ``python -O``). Validation may be expensive, so you may want to
        disable it once a model is working.

        Args:
            value (bool): Whether to enable validation.
        """
        if value not in [True, False]:
            raise ValueError
        Distribution._validate_args = value
    
    def __init__(self, 
        batch_shape: torch.Size = torch.Size(),
        event_shape: torch.Size = torch.Size(),
        validate_args: Optional[Any] = None):

        self._batch_shape = batch_shape
        self._event_shape = event_shape

        if validate_args is None:
            self._validate_args = validate_args
        if self._validate_args:
            arg_constraints: Dict = {}
            try:
                arg_constraints = self.arg_constraints
            except NotImplementedError:
                warnings.warn(f'{self.__class__} does not define `arg_constraints`. ' +
                              'Please set `arg_constraints = {}` or initialize the distribution ' +
                              'with `validate_args=False` to turn off validation.')
        for param, contraint in arg_constraints.items():
            pass
    
    @torch.jit.export
    @property
    def arg_constraits(self) -> Dict[str, constraints.Constraint]:
        raise NotImplementedError


'''



from typing import Tuple
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


# TODO: optimize this terrible function
@torch.jit.script
def broadcast_tensors(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    shape = _broadcast_shape_over_list(tensor_list)
    return [torch.broadcast_to(tensor, shape) for tensor in tensor_list]


@torch.jit.script
def _standard_normal(shape: List[int], dtype: torch.dtype, device: torch.device):
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(torch.zeros(shape, dtype=dtype, device=device),
                            torch.ones(shape, dtype=dtype, device=device))
    return torch.empty(shape, dtype=dtype, device=device).normal_()



@torch.jit.script
class Normal:

    # torch.Size not implemented
    def __init__(self, 
            loc: torch.Tensor,
            scale: torch.Tensor
            ):
        

        # yea so like this is fine because its an init and who cares
        if isinstance(loc, Number):
            loc = torch.tensor(loc)
        if isinstance(scale, Number):
            scale = torch.tensor(scale)
        
        self.loc, self.scale = broadcast_tensors([loc, scale])
        batch_shape = self.loc.size()
        event_shape: List[int] = []

        # Normal init
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        #self._event_shape = event_shape


    @property
    def mean(self):
        return self.loc

    @property
    def variance(self) -> torch.Tensor:
        return self.stddev.pow(2)
    
    @property
    def stddev(self):
        return self.scale


    
    # not doing anything smart yet
    def expand(self, batch_shape: torch.Tensor):
        #new = torch.jit.script(Normal)
        new_loc = self.loc.expand(batch_shape)
        new_scale = self.scale.expand(batch_shape)
        return Normal(new_loc, new_scale) # i can't torch jit this unfortunately... this might be a preformance problem at some point
    
   
    
    def sample(self, sample_shape: Optional[List[int]] = None ):
        if sample_shape is None:
            a: List[int] = []
            sample_shape = a
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def _extended_shape(self, sample_shape: List[int]) -> List[int]:
        return sample_shape + self._batch_shape + self._event_shape
        #return sample_shape + list(self._batch_shape) + self._event_shape
    

    def rsample(self, sample_shape: Optional[List[int]] = None):
        if sample_shape is None:
            a: list[int] = []
            sample_shape = a
        shape = self._extended_shape(sample_shape) # yes this is aids torchscript doesn't support Size()
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value: torch.Tensor):
        # compute the variance
        var = (self.scale ** 2)
        log_scale = self.scale.log()
        #return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - torch.log(torch.sqrt(2 * torch.pi))

    def cdf(self, value):
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / torch.sqrt(2)))
    

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * torch.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x: torch.Tensor, y: torch.Tensor):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

    @property
    def batch_shape(self):
        return self._batch_shape
    
    @property
    def event_shape(self):
        return self._event_shape
    
    


    # Normal methods


if __name__ == "__main__":
    n = torch.jit.script(Normal(0.50, .1))
    pass