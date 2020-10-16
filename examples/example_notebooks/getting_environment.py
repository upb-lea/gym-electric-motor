import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper


class SqdCurrentMonitor:
    """
    Constraint monitor which monitors if the sum of the squared currents
    is bigger than the nominal limit:

    i_sd**2 + i_sq**2 < nominal_limit
    
    Ends the episode and throws a high negative reward, otherwise.
    """

    def __call__(self, state, observed_states, k, physical_system):
        """
        Gets called at each step to check if the agent's trajectory is violating the above constraint
        """
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        sqd_currents = state[self.I_SD_IDX] ** 2 + state[self.I_SQ_IDX] ** 2
        return sqd_currents > 1


class FeatureWrapper(ObservationWrapper):
    """
    Wrapper class which wraps the environment to change its observation. Serves
    the purpose to improve the agent's learning speed.
    
    For this it changes wpsilon in a flattened observation to cos(epsilon) and
    sin(epsilon) to have epsilon's extreme values (-pi,pi) which are angles next
    to each other have close transitions numerically.
    
    Additionally, this wrapper adds a new observation i_sd**2 + i_sq**2. This should
    help the agent to easier detect incoming limit violations.
    """

    def __init__(self, env, epsilon_idx, i_sd_idx, i_sq_idx):
        """
        Changes the observation space to fit the new features
        
        Args:
            env(GEM env): GEM environment to wrap
            epsilon_idx(integer): Epsilon's index in the observation array
            i_sd_idx(integer): I_sd's index in the observation array
            i_sq_idx(integer): I_sq's index in the observation array
        """
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
        """
        Gets called at each return of an observation. Adds the new features to the
        observation and removes original epsilon.
        
        """
        cos_eps = np.cos(observation[self.EPSILON_IDX] * np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX] * np.pi)
        currents_squared = observation[self.I_SQ_IDX]**2 + observation[self.I_SD_IDX]**2
        observation = np.concatenate((observation[:self.EPSILON_IDX],
                                      np.array([cos_eps, sin_eps]),
                                      observation[self.EPSILON_IDX + 1:],
                                      np.array([currents_squared])))
        return observation


def get_env(time_limit=True, gamma=0.99, training=True, callbacks=[]):
    """
    Returns a fully initialized GEM environment for tutorial purposes.
    
    Args:
        time_limit(bool): If true adds an upper limit of 10000 steps to the length of the environment's episodes
        gamma(float): Gamma value for the reward function. Should be the same or higher than the agent's
                      but smaller than 1. Decides how big the negative reward for the limit violation is
        training(bool): If false the reward function is the MAE with a fixed gamma of 0.99 to get comparable test rewards
        callbacks(list): List of callbacks for the environment
    
    """
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
