import numpy as np
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from agents.simple_controllers import Controller
import time
from scipy import signal
from gym_electric_motor.physical_systems.mechanical_loads \
    import ExternalSpeedLoad, ConstantSpeedLoad

'''
This code example presents how the speed load classes can be used to
define a speed profile which the drive will then follow.
This is often useful when prototyping drive controllers that either
control purely electrical drive behavior (e.g. current controllers) or
drive torque with fixed speed requirements (e.g. traction applications, generator operation).

For a more general introduction to GEM, we recommend to have a look at the "_control.py" examples first.
'''


# We will have a look at a current control scenario here
const_sub_gen = [rg.ConstReferenceGenerator(reference_state='i', reference_value=0.1),  # operation at 10 % of the current limit
                 rg.ConstReferenceGenerator(reference_state='i', reference_value=0.9),  # operation at 90 % of the current limit
                 rg.ConstReferenceGenerator(reference_state='i', reference_value=1.1),  # operation at 110 % of the current limit,
                 ]

# since the case of 110 % current may lead to limit violation, we only assign a small probability of 2 % to it
const_switch_gen = rg.SwitchedReferenceGenerator(const_sub_gen,
                                                 p=[0.49, 0.49, 0.02],
                                                 super_episode_length=(1000, 2000))

# The ExternalSpeedLoad class allows to pass an arbitrary function of time which will then dictate the speed profile.
# As shown here it can also contain more parameters. Some examples:
# Parameterizable sine oscillation
sinus_lambda = (lambda t, frequency, amplitude, bias: amplitude * np.sin(2 * np.pi * frequency * t) + bias)
# Constant speed
constant_lambda = (lambda t, value: value)
# Parameterizable triangle oscillation
triangle_lambda = (lambda t, amplitude, frequency, bias: amplitude * signal.sawtooth(2 * np.pi * frequency * t,
                                                                                     width=0.5) + bias)
# Parameterizable sawtooth oscillation
saw_lambda = (lambda t, amplitude, frequency, bias: amplitude * signal.sawtooth(2 * np.pi * frequency * t,
                                                                                width=0.9) + bias)


# usage of a random load initializer is only recommended for the
# ConstantSpeedLoad, due to the already given profile by an ExternalSpeedLoad
load_init = {'random_init': 'uniform'},

# External speed profiles can be given by an ExternalSpeedLoad,
# inital value is given by bias of the profile
sampling_time = 1e-4

if __name__ == '__main__':
    # Create the environment
    env = gem.make(
        'DcSeriesCont-v1',
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        tau=sampling_time,
        reference_generator=const_switch_gen,
        visualization=MotorDashboard(plots=['omega', 'reward', 'i'],
                                     dark_mode=False),

        # define which load to use, feel free to try these examples:

        # using ExternalSpeedLoad:
        #load=ExternalSpeedLoad(speed_profile=sinus_lambda, tau=sampling_time, amplitude=10, frequency=2, bias=4)
        load=ExternalSpeedLoad(speed_profile=saw_lambda, tau=sampling_time, amplitude=40, frequency=5, bias=40)
        #load=ExternalSpeedLoad(speed_profile=constant_lambda, value=3, load_initializer={'random_init': 'uniform', 'interval': [[20, 80]]})
        #(ExternalSpeedLoad will not use the passed initializer and print a corresponding warning instead)

        # using ConstantSpeedLoad (this class also accepts an initializer):
        #load=ConstantSpeedLoad(omega_fixed=42, load_initializer={'random_init': 'uniform', 'interval': [[20, 80]]})
    )

    # After the setup is done, we are ready to simulate the environment
    # We make use of a standard PI current controller
    controller = Controller.make('pi_controller', env)

    episode_duration = 0.2  # episode duration in seconds
    steps_per_episode = int(episode_duration / sampling_time)
    nb_episodes = 20

    for eps in range(nb_episodes):
        state, reference = env.reset()
        start = time.time()
        cum_rew = 0

        for i in range(5000):
            env.render()
            action = controller.control(state, reference)
            (state, reference), reward, done, _ = env.step(action)

            if done:
                state, _ = env.reset()
                controller.reset()
            cum_rew += reward

        print(cum_rew)
