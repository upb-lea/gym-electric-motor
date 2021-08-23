import numpy as np
import gym


def state_dict_to_state_array(state_dict, state_array, state_names):
    """
    Mapping of a passed state dictionary to a fitting state array.

    This function is mainly used in the initialization phase to map a dictionary of passed state_name, state_value pairs
    to a numpy state array with the entries of the state_dict at the corresponding places of the state_names.

    Args:
        state_dict(dict): Dictionary containing pairs of state_name, state_values for the state_array
        state_array(iterable): Array into which the state_dict entries shall be passed
        state_names(list/ndarray(str)): List of the state names.
    """
    state_dict = dict((key.lower(), v) for key, v in state_dict.items())
    assert all(key in state_names for key in state_dict.keys()), f'A state name in {state_dict.keys()} is invalid.'
    for ind, key in enumerate(state_names):
        try:
            state_array[ind] = state_dict[key]
        except KeyError:
            pass


def set_state_array(input_values, state_names):
    """
    Setting of the input values to a valid state array with the shape of the physical systems state.

    The input values can be passed as dict with state_name: value pairs or as list  / ndarray. In the latter case the
    shape of the list has to fit the state_names shape and the list will just be returned as array. If a float is
    passed as input value, then this value will be set onto all positions of the state array equally.

    Args:
        input_values(dict(float) / list(float) / ndarray(float) / float): Values to be set onto the state array.
        state_names(list(str)): List containing the state names of the physical system.

    Returns:
        An initialized state array with all values passed in input values set onto the corresponding position in
        the state_names and zero otherwise.
    """

    if type(input_values) is dict:
        state_array = np.zeros_like(state_names, dtype=float)
        state_dict_to_state_array(input_values, state_array, state_names)
    elif type(input_values) is np.ndarray:
        assert len(input_values) == len(state_names)
        state_array = input_values
    elif type(input_values) is list:
        assert len(input_values) == len(state_names)
        state_array = np.array(input_values)
    elif type(input_values) is float or type(input_values) is int:
        state_array = input_values * np.ones_like(state_names, dtype=float)
    else:
        raise Exception('Incorrect type for the input values.')
    return state_array


def initialize(base_class, arg, default_class, default_args):
    if arg is None:
        return default_class(**default_args)
    elif isinstance(arg, base_class):
        return arg
    elif type(arg) is str:
        return _registry[base_class][arg]()
    elif type(arg) is dict:
        default_args.update(arg)
        return default_class(**default_args)


def instantiate(superclass, instance, **kwargs):
    """
    Instantiation of an instance that inherits from the passed superclass.

    The instance can be passed as a key-string, a class pointer or an already instantiated object. In the latter case
    the same object will be simply returned. If a string is passed the corresponding class will be taken from the
    registry and instantiated with the given kwargs. If a class pointer is passed, then the class is instantiated with
    with given kwargs, directly.

    Args:
        superclass(class): Superclass pointer for registry access
        instance(str, class, object): Instance to instantiate
        kwargs: Arguments for the instantiation of the object

    Returns:
        An instantiated object.
    """
    if type(instance) is type and issubclass(instance, superclass):
        return instance(**kwargs)
    elif isinstance(instance, superclass):
        return instance
    elif type(instance) is str:
        return make_module(superclass, instance, **kwargs)
    else:
        raise Exception('Instantiation Error.')


# Registry dictionary that stores the keys to instantiate the components with the keystrings
_registry = {}


def make_module(superclass, keystring, **kwargs):
    """
    Instantiation by an object that is specified by the key-string an its superclass from the registry.

    Args:
        superclass(class): Superclass pointer for registry access
        keystring(str): String to access the class pointer in the registry.
        kwargs: Arguments for the instantiation of the object.

    Returns:
        An instantiated object.
    """
    try:
        return _registry[superclass][keystring](**kwargs)
    except KeyError:
        raise Exception(f'Key {keystring} or baseclass {superclass.__name__} not found in the registry.')


def register_superclass(superclass):
    """
    Method to register a new superclass that can contain several key-strings for instantiation in the registry.

    Basically, all superclasses in GEM are already registered like the Physical Systems, Reference Generators, ...
    """
    _registry[superclass] = {}


def register_class(subclass, superclass, keystring):
    """
    Method to register a new class with a key-string into the registry of the superclass
    to be instantiable with the key-string.
    """
    _registry[superclass][keystring] = subclass


def update_parameter_dict(source_dict, update_dict, copy=True):
    """Merges two dictionaries (source and update) together.

    It is similar to pythons dict.update() method. Furthermore, it assures that all keys in the update dictionary are
    already present in the source dictionary. Otherwise a KeyError is thrown.

    Arguments:
          source_dict(dict): Source dictionary to be updated.
          update_dict(dict): The new dictionary with the entries to update the source dict.
          copy(bool): Flag, if the source dictionary shall be copied before updating. (Default True)
    Returns:
        dict: The updated source dictionary.
    Exceptions:
        KeyError: Thrown, if a key in the update dict is not available in the source dict.
    """
    source_keys = source_dict.keys()
    for key in update_dict.keys():
        if key not in source_keys:
            raise KeyError(f'Cannot update_dict the source_dict. The key "{key}" is not available.')
    new_dict = source_dict.copy() if copy else source_dict
    new_dict.update(update_dict)
    return new_dict


#: Short notation for the gym.make call to avoid the necessary import of gym when making environments.
make = gym.make
