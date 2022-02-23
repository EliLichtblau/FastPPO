"""
Torch script does not support inheritance, so to see the inherited method
look in __init__

"""



r"""
The following constraints are implemented:

- ``constraints.boolean`` X
- ``constraints.cat`` X
- ``constraints.corr_cholesky`` X
- ``constraints.dependent`` X
- ``constraints.greater_than(lower_bound)`` X
- ``constraints.greater_than_eq(lower_bound)`` X
- ``constraints.independent(constraint, reinterpreted_batch_ndims)`` X
- ``constraints.integer_interval(lower_bound, upper_bound)`` X
- ``constraints.interval(lower_bound, upper_bound)`` X
- ``constraints.less_than(upper_bound)`` X
- ``constraints.lower_cholesky`` X
- ``constraints.lower_triangular`` X
- ``constraints.multinomial`` X
- ``constraints.nonnegative_integer`` X
- ``constraints.one_hot`` X
- ``constraints.positive_definite`` X
- ``constraints.positive_integer`` X
- ``constraints.positive`` X 
- ``constraints.real_vector`` X
- ``constraints.real`` X
- ``constraints.simplex`` X
- ``constraints.stack`` X
- ``constraints.unit_interval`` X
"""
import torch

__all__ = [
    'Constraint',
    'boolean',
    'cat',
    'corr_cholesky',
    'dependent',
    'dependent_property',
    'greater_than',
    'greater_than_eq',
    'independent',
    'integer_interval',
    'interval',
    'half_open_interval',
    'is_dependent',
    'less_than',
    'lower_cholesky',
    'lower_triangular',
    'multinomial',
    'nonnegative_integer',
    'positive',
    'positive_definite',
    'positive_integer',
    'real',
    'real_vector',
    'simplex',
    'stack',
    'unit_interval',
]




'''


'''


# Yes this is an ABC
#@torch.jit.script
class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a variable is valid,
    e.g. within which a variable can be optimized.

    Attributes:
        is_discrete (bool): Whether constrained space is discrete.
            Defaults to False.
        event_dim (int): Number of rightmost dimensions that together define
            an event. The :meth:`check` method will remove this many dimensions
            when computing validity.
    """
    is_discrete = False  # Default to continuous.
    event_dim = 0  # Default to univariate.

    def __init__(self):
        self.name = "Constraint" # dunder methods are weird with jit
        raise NotImplementedError


    def check(self, value):
        """
        Returns a byte tensor of ``sample_shape + batch_shape`` indicating
        whether each event in value satisfies this constraint.
        """
        raise NotImplementedError

    def __repr__(self):
        return self.name + '()'


#breakpoint()

#@torch.jit.script
class _Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    These variables obey no simple coordinate-wise constraints.

    Args:
    is_discrete (bool): Optional value of ``.is_discrete`` in case this
        can be computed statically. If not provided, access to the
        ``.is_discrete`` attribute will raise a NotImplementedError.
    event_dim (int): Optional value of ``.event_dim`` in case this
        can be computed statically. If not provided, access to the
        ``.event_dim`` attribute will raise a NotImplementedError.
    """
    def __init__(self, is_discrete=NotImplemented, event_dim=NotImplemented):
        self._is_discrete = is_discrete
        self._event_dim = event_dim
        super().__init__()



@torch.jit.export
def inherit(_superClass):
    #import inspect
    #import re
    #import textwrap
    #find_super_regex = re.compile(r"super\(.*\)\.__init__\(.*\)\s*\n")
    #function_def_regex = re.compile(r"def\(.*\):")
    @torch.jit.export
    def decorator(_subClass):


        sub_init = _subClass.__init__

       
        

        #setattr(_subClass, "new_init", sub_init)
        @torch.jit.export
        def new_init(self: object):
            #print(dir(_superClass.__init__(self)))
            #setattr(_subClass, "someField", 10)
            _superClass.__init__(self)
            #setattr(_subClass, "tryme", 10)
            sub_init(self)
            #dummy_init(self)
            self.tryme = 10
            
            
            #self.someField = 10
            #Inherit.__init__(self)
            #self.killme = 100
            #_subClass.__init__(self)
            
        #_subClass.__dict__[new_init.__name__] = new_init
        #setattr(_subClass, "someField", 10)
        #setattr(_subClass, _superClass.__init__.__name__, _superClass.__init__)
        
        setattr(_subClass, new_init.__name__, new_init)
        #print(f"PRINTING SUBCLASS: {_subClass}")
        _subClass.__init__ = new_init
        #print(inspect.getsource(_subClass.__init__))

        #print(_subClass.__init__)
        # Add methods from superClass to subClass
        #super = _superClass()
        #super_class_methods = set(dir(_superClass))
        #super_function_pointers = vars(_superClass)

        #sub_class_methods = set(dir(_subClass))
        #non_overriden_methods = super_class_methods - sub_class_methods

        #for method in non_overriden_methods:
        #    setattr(_subClass, method, super_function_pointers[method])

        # Add fields by replacing init with first a call to super init with pass self
        
        
        


        return _subClass
    
    return decorator



'''
@torch.jit.script
class Dummy:
    @torch.jit.export
    def __init__(self):
        #super(Dummy, self).__init__()
        #print(f"self: {self}")
        self.someField: int = 10


    @torch.jit.export
    def method1(self):
        print("method1 called")
    @torch.jit.export
    def method2(self):
        print(self.someField)
'''


'''

@torch.jit.script
class Dummy:
    @torch.jit.export
    def __init__(self):
        self.someField: int = 10
    

    @torch.jit.export
    def method1(self):
        print("method 1")
    



'''




'''
#@torch.jit.export
@torch.jit.script
@inherit(Dummy)
class Inherit:
    @torch.jit.export
    def __init__(self):
        #Dummy.__init__(self)
        self.inside_init: int = 20
        #self.bs()
    @torch.jit.export
    def longfunc(self):
        r = torch.randn((100, 100))
        w = torch.randn((100, 100))
        return r* w

#i = torch.jit.script(Inherit())
i = Inherit()

'''

'''

@torch.jit.script
class Inherit:
    @torch.jit.export
    def __init__(self):
        self.x = 5

    @torch.jit.export
    def randommethod(self, x):
        return x.flatten()

i = torch.jit.script(Inherit())
breakpoint()


'''


import ast
import inspect
import astdump

def base_init(self):
    self.baseVar = 1

class SubClass:
    def __init__(self):
        base_init(self)
        self.a = 2
        self.b = 3

    def basemethod(self):
        return 5




import dis
print(dis.dis(SubClass.__init__))

print(astdump.indented(ast.parse(inspect.getsource(SubClass))))
