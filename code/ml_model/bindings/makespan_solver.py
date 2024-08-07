# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _makespan_solver
else:
    import _makespan_solver

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _makespan_solver.delete_SwigPyIterator

    def value(self):
        return _makespan_solver.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _makespan_solver.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _makespan_solver.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _makespan_solver.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _makespan_solver.SwigPyIterator_equal(self, x)

    def copy(self):
        return _makespan_solver.SwigPyIterator_copy(self)

    def next(self):
        return _makespan_solver.SwigPyIterator_next(self)

    def __next__(self):
        return _makespan_solver.SwigPyIterator___next__(self)

    def previous(self):
        return _makespan_solver.SwigPyIterator_previous(self)

    def advance(self, n):
        return _makespan_solver.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _makespan_solver.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _makespan_solver.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _makespan_solver.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _makespan_solver.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _makespan_solver.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _makespan_solver.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _makespan_solver:
_makespan_solver.SwigPyIterator_swigregister(SwigPyIterator)

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _makespan_solver.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _makespan_solver.IntVector___nonzero__(self)

    def __bool__(self):
        return _makespan_solver.IntVector___bool__(self)

    def __len__(self):
        return _makespan_solver.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _makespan_solver.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _makespan_solver.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _makespan_solver.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _makespan_solver.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _makespan_solver.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _makespan_solver.IntVector___setitem__(self, *args)

    def pop(self):
        return _makespan_solver.IntVector_pop(self)

    def append(self, x):
        return _makespan_solver.IntVector_append(self, x)

    def empty(self):
        return _makespan_solver.IntVector_empty(self)

    def size(self):
        return _makespan_solver.IntVector_size(self)

    def swap(self, v):
        return _makespan_solver.IntVector_swap(self, v)

    def begin(self):
        return _makespan_solver.IntVector_begin(self)

    def end(self):
        return _makespan_solver.IntVector_end(self)

    def rbegin(self):
        return _makespan_solver.IntVector_rbegin(self)

    def rend(self):
        return _makespan_solver.IntVector_rend(self)

    def clear(self):
        return _makespan_solver.IntVector_clear(self)

    def get_allocator(self):
        return _makespan_solver.IntVector_get_allocator(self)

    def pop_back(self):
        return _makespan_solver.IntVector_pop_back(self)

    def erase(self, *args):
        return _makespan_solver.IntVector_erase(self, *args)

    def __init__(self, *args):
        _makespan_solver.IntVector_swiginit(self, _makespan_solver.new_IntVector(*args))

    def push_back(self, x):
        return _makespan_solver.IntVector_push_back(self, x)

    def front(self):
        return _makespan_solver.IntVector_front(self)

    def back(self):
        return _makespan_solver.IntVector_back(self)

    def assign(self, n, x):
        return _makespan_solver.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _makespan_solver.IntVector_resize(self, *args)

    def insert(self, *args):
        return _makespan_solver.IntVector_insert(self, *args)

    def reserve(self, n):
        return _makespan_solver.IntVector_reserve(self, n)

    def capacity(self):
        return _makespan_solver.IntVector_capacity(self)
    __swig_destroy__ = _makespan_solver.delete_IntVector

# Register IntVector in _makespan_solver:
_makespan_solver.IntVector_swigregister(IntVector)

class DagSubtaskVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _makespan_solver.DagSubtaskVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _makespan_solver.DagSubtaskVector___nonzero__(self)

    def __bool__(self):
        return _makespan_solver.DagSubtaskVector___bool__(self)

    def __len__(self):
        return _makespan_solver.DagSubtaskVector___len__(self)

    def __getslice__(self, i, j):
        return _makespan_solver.DagSubtaskVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _makespan_solver.DagSubtaskVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _makespan_solver.DagSubtaskVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _makespan_solver.DagSubtaskVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _makespan_solver.DagSubtaskVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _makespan_solver.DagSubtaskVector___setitem__(self, *args)

    def pop(self):
        return _makespan_solver.DagSubtaskVector_pop(self)

    def append(self, x):
        return _makespan_solver.DagSubtaskVector_append(self, x)

    def empty(self):
        return _makespan_solver.DagSubtaskVector_empty(self)

    def size(self):
        return _makespan_solver.DagSubtaskVector_size(self)

    def swap(self, v):
        return _makespan_solver.DagSubtaskVector_swap(self, v)

    def begin(self):
        return _makespan_solver.DagSubtaskVector_begin(self)

    def end(self):
        return _makespan_solver.DagSubtaskVector_end(self)

    def rbegin(self):
        return _makespan_solver.DagSubtaskVector_rbegin(self)

    def rend(self):
        return _makespan_solver.DagSubtaskVector_rend(self)

    def clear(self):
        return _makespan_solver.DagSubtaskVector_clear(self)

    def get_allocator(self):
        return _makespan_solver.DagSubtaskVector_get_allocator(self)

    def pop_back(self):
        return _makespan_solver.DagSubtaskVector_pop_back(self)

    def erase(self, *args):
        return _makespan_solver.DagSubtaskVector_erase(self, *args)

    def __init__(self, *args):
        _makespan_solver.DagSubtaskVector_swiginit(self, _makespan_solver.new_DagSubtaskVector(*args))

    def push_back(self, x):
        return _makespan_solver.DagSubtaskVector_push_back(self, x)

    def front(self):
        return _makespan_solver.DagSubtaskVector_front(self)

    def back(self):
        return _makespan_solver.DagSubtaskVector_back(self)

    def assign(self, n, x):
        return _makespan_solver.DagSubtaskVector_assign(self, n, x)

    def resize(self, *args):
        return _makespan_solver.DagSubtaskVector_resize(self, *args)

    def insert(self, *args):
        return _makespan_solver.DagSubtaskVector_insert(self, *args)

    def reserve(self, n):
        return _makespan_solver.DagSubtaskVector_reserve(self, n)

    def capacity(self):
        return _makespan_solver.DagSubtaskVector_capacity(self)
    __swig_destroy__ = _makespan_solver.delete_DagSubtaskVector

# Register DagSubtaskVector in _makespan_solver:
_makespan_solver.DagSubtaskVector_swigregister(DagSubtaskVector)

class IntList(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _makespan_solver.IntList_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _makespan_solver.IntList___nonzero__(self)

    def __bool__(self):
        return _makespan_solver.IntList___bool__(self)

    def __len__(self):
        return _makespan_solver.IntList___len__(self)

    def __getslice__(self, i, j):
        return _makespan_solver.IntList___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _makespan_solver.IntList___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _makespan_solver.IntList___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _makespan_solver.IntList___delitem__(self, *args)

    def __getitem__(self, *args):
        return _makespan_solver.IntList___getitem__(self, *args)

    def __setitem__(self, *args):
        return _makespan_solver.IntList___setitem__(self, *args)

    def pop(self):
        return _makespan_solver.IntList_pop(self)

    def append(self, x):
        return _makespan_solver.IntList_append(self, x)

    def empty(self):
        return _makespan_solver.IntList_empty(self)

    def size(self):
        return _makespan_solver.IntList_size(self)

    def swap(self, v):
        return _makespan_solver.IntList_swap(self, v)

    def begin(self):
        return _makespan_solver.IntList_begin(self)

    def end(self):
        return _makespan_solver.IntList_end(self)

    def rbegin(self):
        return _makespan_solver.IntList_rbegin(self)

    def rend(self):
        return _makespan_solver.IntList_rend(self)

    def clear(self):
        return _makespan_solver.IntList_clear(self)

    def get_allocator(self):
        return _makespan_solver.IntList_get_allocator(self)

    def pop_back(self):
        return _makespan_solver.IntList_pop_back(self)

    def erase(self, *args):
        return _makespan_solver.IntList_erase(self, *args)

    def __init__(self, *args):
        _makespan_solver.IntList_swiginit(self, _makespan_solver.new_IntList(*args))

    def push_back(self, x):
        return _makespan_solver.IntList_push_back(self, x)

    def front(self):
        return _makespan_solver.IntList_front(self)

    def back(self):
        return _makespan_solver.IntList_back(self)

    def assign(self, n, x):
        return _makespan_solver.IntList_assign(self, n, x)

    def resize(self, *args):
        return _makespan_solver.IntList_resize(self, *args)

    def insert(self, *args):
        return _makespan_solver.IntList_insert(self, *args)

    def pop_front(self):
        return _makespan_solver.IntList_pop_front(self)

    def push_front(self, x):
        return _makespan_solver.IntList_push_front(self, x)

    def reverse(self):
        return _makespan_solver.IntList_reverse(self)
    __swig_destroy__ = _makespan_solver.delete_IntList

# Register IntList in _makespan_solver:
_makespan_solver.IntList_swigregister(IntList)

class DagSubtask(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, *args):
        _makespan_solver.DagSubtask_swiginit(self, _makespan_solver.new_DagSubtask(*args))
    id = property(_makespan_solver.DagSubtask_id_get, _makespan_solver.DagSubtask_id_set)
    wcet = property(_makespan_solver.DagSubtask_wcet_get, _makespan_solver.DagSubtask_wcet_set)
    priority = property(_makespan_solver.DagSubtask_priority_get, _makespan_solver.DagSubtask_priority_set)
    inDependencies = property(_makespan_solver.DagSubtask_inDependencies_get, _makespan_solver.DagSubtask_inDependencies_set)
    outDependencies = property(_makespan_solver.DagSubtask_outDependencies_get, _makespan_solver.DagSubtask_outDependencies_set)
    __swig_destroy__ = _makespan_solver.delete_DagSubtask

# Register DagSubtask in _makespan_solver:
_makespan_solver.DagSubtask_swigregister(DagSubtask)

class MakespanSolver(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, numberOfCores=4):
        _makespan_solver.MakespanSolver_swiginit(self, _makespan_solver.new_MakespanSolver(numberOfCores))

    def computeMakespan(self, priorityList, dagTask):
        return _makespan_solver.MakespanSolver_computeMakespan(self, priorityList, dagTask)
    __swig_destroy__ = _makespan_solver.delete_MakespanSolver

# Register MakespanSolver in _makespan_solver:
_makespan_solver.MakespanSolver_swigregister(MakespanSolver)



