Utils
#####

The Utils module contains a wrapper for the discrete externally excited motor, an simple Euler integrator and an
optionparsing to specify parameter, e.g. motor parameter.
The wrapper is needed that the motor fits to the kers-rl DQN.
The Euler integrator has go the same interface as the dopri5 integrator from the scipy.integrate.ode package.


.. automodule:: gym_electric_motor.envs.utils
    :members:

