import numpy as np
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from agents.simple_controllers import Controller
import time
from scipy import signal
from gym_electric_motor.physical_systems.mechanical_loads \
    import ExternalSpeedLoad, ConstantSpeedLoad

# defining different reference generators
const_sub_gen = [rg.ConstReferenceGenerator(reference_value=0.20),
                 rg.ConstReferenceGenerator(reference_value=0.25),
                 rg.ConstReferenceGenerator(reference_value=0.30),
                 rg.ConstReferenceGenerator(reference_value=0.35),
                 rg.ConstReferenceGenerator(reference_value=0.40)]
const_switch_gen = rg.SwitchedReferenceGenerator(const_sub_gen,
                                                 super_episode_length=(1000, 2000))

# possible speed profiles
sinus_lambda = (lambda t,f,amp, b: amp * np.sin(2*np.pi*f*t) + b)
constant_lambda = (lambda value: value)
triangle_lambda = (lambda t, amp, f, b: amp * signal.sawtooth(2 * np.pi * f * t,
                                                             width=0.5) + b)
saw_lambda = (lambda t, amp, f, b: amp * signal.sawtooth(2 * np.pi * f * t) + b)

# usage of a random load initializer is only recommended for the
# ConstantSpeedLoad, due to the already given profile by an ExternalSpeedLoad
load_init = {'random_init': 'uniform'},

# defining a ConstandSpeedLoad
const_load = ConstantSpeedLoad(omega_fixed=42)
# External speed profiles can be given by an ExternalSpeedLoad,
# inital value is given by bias of the profile
ext_load = ExternalSpeedLoad(speed_profile=sinus_lambda, amp=10, f=2, b=42)

if __name__ == '__main__':
    env = gem.make(
        'DcSeriesCont-v1',
        # Pass an instance
        visualization=MotorDashboard(plots=['omega', 'reward', 'i'],
                                     dark_mode=False),
        motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),
        # Take standard class and pass parameters (Load)
        load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.06),
        # Pass a string (with extra parameters)
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        # Pass a Class with extra parameters
        load=ext_load, # const_load
        tau=1e-4,

        # define state initializer
        motor_initializer={'random_init': 'uniform'},
        # define reference generator
        reference_generator=const_switch_gen,
    )

    # works for 'on_off'(disc),three_point(disc),'p_controller', 'cascaded pi' too
    controller = Controller.make('pi_controller', env)
    for eps in range(10):
        state, reference = env.reset()
        start = time.time()
        cum_rew = 0
        for i in range(5000):
            env.render()
            action = controller.control(state, reference)
            (state, reference), reward, done, _ = env.step(action)

            if done:
                state, _ = env.reset()
            cum_rew += reward
        print(cum_rew)