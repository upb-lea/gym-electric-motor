import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.constraint_monitor import ConstraintMonitor
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import ConstantSpeedLoad

from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper


class SqdCurrentMonitor:
    """
    monitor for squared currents:

    i_sd**2 + i_sq**2 < 1.5 * nominal_limit
    """

    def __call__(self, state, observed_states, k, physical_system):
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        # normalize to limit_values, since state is normalized
        nominal_values = physical_system.nominal_state / abs(
            physical_system.limits)
        limits = 1.5 * nominal_values
        # calculating squared currents as observed measure
        sqd_currents = state[self.I_SD_IDX] ** 2 + state[self.I_SQ_IDX] ** 2

        return (sqd_currents > limits[self.I_SD_IDX] or sqd_currents > limits[
            self.I_SQ_IDX])


class EpsilonWrapper(ObservationWrapper):
    """
    Changes Epsilon in a flattened observation to cos(epsilon)
    and sin(epsilon)
    """

    def __init__(self, env, epsilon_idx):
        super(EpsilonWrapper, self).__init__(env)
        self.EPSILON_IDX = epsilon_idx
        new_low = np.concatenate((self.env.observation_space.low[
                                  :self.EPSILON_IDX], np.array([-1.]),
                                  self.env.observation_space.low[
                                  self.EPSILON_IDX:]))
        new_high = np.concatenate((self.env.observation_space.high[
                                   :self.EPSILON_IDX], np.array([1.]),
                                   self.env.observation_space.high[
                                   self.EPSILON_IDX:]))

        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        observation = np.concatenate((observation[:self.EPSILON_IDX],
                                      np.array([cos_eps, sin_eps]),
                                      observation[self.EPSILON_IDX + 1:]))
        return observation


def set_env(time_limit=True):
    # define motor arguments
    motor_parameter = dict(p=3,  # [p] = 1, nb of pole pairs
                           r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                           l_d=0.37e-3,  # [l_d] = H, d-axis inductance
                           l_q=1.2e-3,  # [l_q] = H, q-axis inductance
                           psi_p=65.65e-3,  # [psi_p] = Vs, magnetic flux of the permanent magnet
                           )
    u_sup = 350
    nominal_values=dict(omega=4000*2*np.pi/60,
                        i=230,
                        u=u_sup
                        )
    limit_values=nominal_values.copy()

    motor_init = {'random_init': 'uniform'}
    load_initializer = {'random_init': 'uniform'}

    const_random_load = ConstantSpeedLoad(load_initializer=load_initializer)
    q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
    d_generator = WienerProcessReferenceGenerator(reference_state='i_sd')
    rg = MultipleReferenceGenerator([q_generator, d_generator])
    tau = 1e-5
    gamma = 0.99
    max_eps_steps = 10000

    # creating gem environment
    env = gem.make(  # define a PMSM with discrete action space
        "PMSMDisc-v1",
        # visualize the results
        visualization=MotorDashboard(plots=['i_sq', 'i_sd', 'reward']),
        # parameterize the PMSM and update limitations
        motor_parameter=motor_parameter,
        limit_values=limit_values, nominal_values=nominal_values,
        # define the random initialisation for load and motor
        load='ConstSpeedLoad',
        load_initializer={'random_init': 'uniform', },
        motor_initializer={'random_init': 'uniform', },
        # define the duration of one sampling step
        tau=tau, u_sup=u_sup,
        # turn off terminations via limit violation, parameterize the rew-fct
        reward_function=gem.reward_functions.WeightedSumOfErrors(
            observed_states=['i_sq', 'i_sd'],
            reward_weights={'i_sq': 1, 'i_sd': 1},
            constraint_monitor=SqdCurrentMonitor(),
            gamma=gamma,
            reward_power=1),
        reference_generator=rg, ode_solver='euler',
    )

    # appling wrappers and modifying environment
    env.action_space = Discrete(7)
    eps_idx = env.physical_system.state_names.index('epsilon')
    if time_limit:
        gem_env = TimeLimit(EpsilonWrapper(FlattenObservation(env), eps_idx),
                            max_eps_steps)
    else:
        gem_env = EpsilonWrapper(FlattenObservation(env), eps_idx)
    return gem_env
