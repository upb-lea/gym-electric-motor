import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
import sys
sys.path.append('..')
from classic_controllers.simple_controllers import Controller
import time

'''
This code example presents how the initializer interface can be used to sample random initial states for the drive.
This is important when using e.g. reinforcement learning, because random initialization allows for a better
exploration of the state space (so called "exploring starts").
Initializers can be applied to electric motor state (which is also decisive for the initial torque) and
mechanical load (which sets initial drive speed).

For a more general introduction to GEM, we recommend to have a look at the "_control.py" examples first.
'''


# Initializers use the state names that are present for the used motor:

# initializer for a specific current; e.g. DC series motor ('DcSeriesCont-v1' / 'DcSeriesDisc-v1')
dc_series_init = {'states': {'i': 12}}

# initializer for a specific current and position; e.g. permanent magnet synchronous motor
pmsm_init = {
    'states': {
        'i_sd': -36.0,
        'i_sq': 55.0,
        'epsilon': 3.0
    }
}

# initializer for a random initial current with gaussian distribution, parameterized with mu=25 and sigma=10
gaussian_init = {
    'random_init': 'gaussian',
    'random_params': (25, 0.1),
    'states': {'i': 0}
}

# initializer for a ranom initial speed with uniform distribution within the interval omega=60 to omega=80
uniform_init = {
    'random_init': 'uniform',
    'interval': [[60, 80]],
    'states': {'omega': 0}
}

# initializer for a specific speed
load_init = {'states': {'omega': 20}}

if __name__ == '__main__':
    env = gem.make(
            'Cont-CC-SeriesDc-v0',
            visualization=MotorDashboard(state_plots=['omega', 'i']),
            motor=dict(motor_parameter=dict(j_rotor=0.001), motor_initializer=gaussian_init),
            load=dict(load_parameter=dict(a=0, b=0.1, c=0, j_load=0.001), load_initializer=uniform_init),
            ode_solver='scipy.solve_ivp',
            reference_generator=rg.SwitchedReferenceGenerator(
                sub_generators=[
                    rg.SinusoidalReferenceGenerator(reference_state='omega'),
                    rg.WienerProcessReferenceGenerator(reference_state='omega'),
                    rg.StepReferenceGenerator(reference_state='omega')
                ],
                p=[0.2, 0.6, 0.2],
                super_episode_length=(1000, 10000)
            ),
            constraints=(),
        )
    start = time.time()
    cum_rew = 0

    for j in range(10):
        state, reference = env.reset()

        # Print the initial states:
        denorm_state = state * env.limits
        print(f"Initial speed: {denorm_state[0]:3.2f} 1/s")
        print(f"Initial current: {denorm_state[2]:3.2f} A")
        # We should be able to see that the initial state fits the used initializers
        # Here we should have omega in the interval [60 1/s, 80 1/s] and current closely around 25 A
        print()

        for i in range(5000):
            env.render()
            action = env.action_space.sample()
            (state, reference), reward, done, _ = env.step(action)

            if done:
                break




