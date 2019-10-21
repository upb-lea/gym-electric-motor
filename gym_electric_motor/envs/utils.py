from gym.core import ActionWrapper, Wrapper
import numpy as np
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class FlattenActionWrapper(ActionWrapper):
    """
    Wrapper for the discrete action externally excited motor environment to make the actions fit to keras-rl DQN agent.
    """
    def __init__(self, env):
        super().__init__(env)

        # Number of actions of the armature circuit converter
        self.nominator = self.env.action_space.nvec[0]

        # Number of actions of the excitation Circuit converter
        self.modulo = self.env.action_space.nvec[1]

    def action(self, action):
        """
        Transform the action from {0,...self.nominator * self.modulo - 1}
        to two dimensional {0, self.nominator - 1} x {0, self.modulo - 1}

        Args:
            action: action from keras-rl
        Returns:
            Action for the Environment
        """
        action1 = action // self.nominator
        action2 = action % self.modulo
        return np.array([action1, action2])

    def reverse_action(self, action):
        """
        Backwards transformation of the action.

        Args:
            action: Action for the environment
        Returns:
            Action for keras-rl
        """
        return self.nominator * action[0] + action[1]


class EulerSolver(object):
    """
    Solves a system of differential equations of first order for a given timestep with linear approximation.

        :math:`x^\prime(t) = f(x(t))`

        :math:`x(t + \\frac{\\tau}{nsteps}) = x(t) + x^\prime(t) * \\frac{\\tau}{nsteps}`

    Interface is a subset of the scipy.ode solver interfaces.
    """

    _f_params = None

    @property
    def state(self):
        return self._state

    @property
    def t(self):
        return self._t

    def __init__(self, system_eq, nsteps=1):
        """
        Args:
            system_eq: Function pointer to the systems differential equation. Must accept an n-dimensional state and
                return the n-dimensional derivative of the state.
            nsteps: Number of cycles to calculate for each iteration. Higher steps make the system more accurate,
                but take also longer to compute.
        """
        self._eq = system_eq
        self._t = 0
        self._state = 0
        self._nsteps = nsteps
        self._integrate = self._integrate_onestep if nsteps == 1 else self._integrate_nsteps

    def set_initial_value(self, x, t):
        """
        Set the state of the system to the desired value x at point in time t.

        Args:
            x: The new state of the system.
            t: The new time of the system.

        """
        self._state = x
        self._t = t

    def set_f_params(self, *args):
        """
        Set further arguments for the systems function call like input quantities.

        Args:
            args: Ordered List of arguments for the next function calls.
        """
        self._f_params = args

    @property
    def integrate(self):
        return self._integrate

    def _integrate_nsteps(self, t):
        """
        Integration method for nsteps > 1

        Args:
            t: Time until the system shall be calculated

        Return:
            The new state of the system.
        """
        tau = (t - self._t) / self._nsteps
        state = self._state
        for _ in range(self._nsteps):
            pass
        self._state = state
        state = state + self._eq(self._t, state, *self._f_params) * tau
        self._t = t
        return self._state

    def _integrate_onestep(self, t):
        """
        Integration method for nsteps = 1. (For faster computation)

        Args:
            t: Time until the system shall be calculated

        Return:
            The new state of the system.
        """
        self._state = self._state + self._eq(self._t, self._state, *self._f_params) * (t - self._t)
        self._t = t
        return self._state


def parse_env_args():
    """
    Option parsing function for the gym motor environments.

    **Options:**
        --env_id                Environment id that is used for the gym.make(env_id) call.

        --env_parameter         Parameter json file path, that contains parameters that can be set in the environments
                                | constructor except load_parameter and motor_parameter.

        --motor_parameter       Parameter json file path that contains the technical motor parameters.
                                | These parameters will be set at env_parameter['motor_parameter'] when returned.

        --load_parameter        Parameter json file path that contains the technical load parameters.
                                | These parameters will be set at env_parameter['load_parameter'] when returned.

    Returns:
         Tuple of the environment_id and the environment constructor parameters including the motor and load parameters.
    """
    argument_parser = ArgumentParser(allow_abbrev=False)
    argument_parser.add_argument('--env_id', type=str, help='Identification id for the environment.')
    argument_parser.add_argument('--env_parameter', type=str, help='File path to the json containing (a subset)'
                                                              ' of the environment parameters. All parameters'
                                                              ' that are not in this file will be set to default')
    argument_parser.add_argument('--load_parameter', type=str, help='File path to the json containing (a subset)'
                                                               ' of the load parameters [a,b,c, J_load], '
                                                               'or the number of the standard parameters'
                                                               'All parameters that are not in this file will be'
                                                               ' set to default')
    argument_parser.add_argument('--motor_parameter', type=str, help='File path to the json containing (a subset)'
                                                                ' of the motor parameters or the number of the standard'
                                                                ' parameters All parameters'
                                                                ' that are not in this file will be set to default')
    arguments, _ = argument_parser.parse_known_args()
    args_dict = arguments.__dict__

    # If an environment parameter file was passed take these parameters, otherwise an empty_dictionary
    if args_dict['env_parameter'] is not None:
        with open(args_dict['env_parameter']) as json_file:
            env_dict = json.load(json_file)
    else:
        env_dict = {}

    # If a load parameter file was passed take these parameters or the number
    if args_dict['load_parameter'] is not None:
        try:
            env_dict['load_parameter'] = int(args_dict['load_parameter'])
        except ValueError:
            with open(args_dict['load_parameter']) as json_file:
                env_dict['load_parameter'] = json.load(json_file)

    # If a motor parameter file was passed take these parameters or the number
    if args_dict['motor_parameter'] is not None:
        try:
            env_dict['motor_parameter'] = int(args_dict['motor_parameter'])
        except ValueError:
            with open(args_dict['motor_parameter']) as json_file:
                env_dict['motor_parameter'] = json.load(json_file)
    return args_dict['env_id'], env_dict
