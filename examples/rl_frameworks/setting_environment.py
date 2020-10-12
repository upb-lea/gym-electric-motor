import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper, Wrapper


# defining the squared current as constraints for the environment
class SqdCurrentMonitor:
    """
    monitor for squared currents:

    i_sd**2 + i_sq**2 < 1.5 * nominal_limit
    """

    def __call__(self, state, observed_states, k, physical_system):
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        sqd_currents = state[self.I_SD_IDX] ** 2 + state[self.I_SQ_IDX] ** 2
        return sqd_currents > 1


class FeatureWrapper(ObservationWrapper):
    """
    Changes Epsilon in a flattened observation to cos(epsilon)
    and sin(epsilon) and adding the squared currents as features
    """

    def __init__(self, env, epsilon_idx, i_sd_idx, i_sq_idx):
        super(FeatureWrapper, self).__init__(env)
        self.EPSILON_IDX = epsilon_idx
        self.I_SQ_IDX = i_sq_idx
        self.I_SD_IDX = i_sd_idx
        new_low = np.concatenate((self.env.observation_space.low[
                                  :self.EPSILON_IDX], np.array([-1.]),
                                  self.env.observation_space.low[
                                  self.EPSILON_IDX:], np.array([0.])))
        new_high = np.concatenate((self.env.observation_space.high[
                                   :self.EPSILON_IDX], np.array([1.]),
                                   self.env.observation_space.high[
                                   self.EPSILON_IDX:],np.array([1.])))

        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        currents_squared = observation[self.I_SQ_IDX]**2 + observation[self.I_SD_IDX]**2
        observation = np.concatenate((observation[:self.EPSILON_IDX],
                                      np.array([cos_eps, sin_eps]),
                                      observation[self.EPSILON_IDX + 1:],
                                      np.array([currents_squared])))
        return observation


class NormalizeObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, term, info = self.env.step(action)
        return obs/np.linalg.norm(obs), rew, term, info

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs/np.linalg.norm(obs)


def set_env(time_limit=True, gamma=0.99, training=True, callbacks=[]):
    # define motor arguments
    motor_parameter = dict(p=3,  # [p] = 1, nb of pole pairs
                           r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                           l_d=0.37e-3,  # [l_d] = H, d-axis inductance
                           l_q=1.2e-3,  # [l_q] = H, q-axis inductance
                           psi_p=65.65e-3,  # [psi_p] = Vs, magnetic flux of the permanent magnet
                           )
    # supply voltage
    u_sup = 350
    # nominal and absolute state limitations
    nominal_values=dict(omega=4000*2*np.pi/60,
                        i=230,
                        u=u_sup
                        )
    limit_values=dict(omega=4000*2*np.pi/60,
                        i=1.5*230,
                        u=u_sup
                        )
    # defining reference-generators
    q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
    d_generator = WienerProcessReferenceGenerator(reference_state='i_sd')
    rg = MultipleReferenceGenerator([q_generator, d_generator])
    # defining sampling interval
    tau = 1e-5
    # defining maximal episode steps
    max_eps_steps = 10000
    
    if training:
        motor_initializer={'random_init': 'uniform', 'interval': [[-230, 230], [-230, 230], [-np.pi, np.pi]]}
        reward_function=gem.reward_functions.WeightedSumOfErrors(
            observed_states=['i_sq', 'i_sd'],
            reward_weights={'i_sq': 10, 'i_sd': 10},
            constraint_monitor=SqdCurrentMonitor(),
            gamma=gamma,
            reward_power=1)
    else:
        motor_initializer = {'random_init': 'gaussian'}
        reward_function=gem.reward_functions.WeightedSumOfErrors(
            observed_states=['i_sq', 'i_sd'],
            reward_weights={'i_sq': 0.5, 'i_sd': 0.5},
            constraint_monitor=SqdCurrentMonitor(),
            gamma=0.99,
            reward_power=1)

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
        motor_initializer=motor_initializer,
        reward_function=reward_function,

        # define the duration of one sampling step
        tau=tau, u_sup=u_sup,
        # turn off terminations via limit violation, parameterize the rew-fct
        reference_generator=rg, ode_solver='euler',
        callbacks=callbacks,
    )

    # applying wrappers and modifying environment
    env.action_space = Discrete(7)
    eps_idx = env.physical_system.state_names.index('epsilon')
    i_sd_idx = env.physical_system.state_names.index('i_sd')
    i_sq_idx = env.physical_system.state_names.index('i_sq')

    if time_limit:
        gem_env = TimeLimit(FeatureWrapper(FlattenObservation(env),
                                           eps_idx, i_sd_idx, i_sq_idx),
                            max_eps_steps)
    else:
        gem_env =FeatureWrapper(FlattenObservation(env),
                                eps_idx, i_sd_idx, i_sq_idx)
    return gem_env
