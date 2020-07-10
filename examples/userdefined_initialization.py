import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard

from agents.simple_controllers import Controller
import time


# initializations for different motor types
# initializer for given constant initial-values
# e.g. for series motor ('DcExtExCont-v1')
dc_series = {'states': {'i': 12}}
# e.g. for synchronous motor ('PMSMCont-v1')
sync_init = {'states': {'i_sq': 5.6, 'i_sd': 6.0, 'epsilon': 3.0}}

# random initial-values in nominal-value-space
gaussian_init = {'random_init': 'gaussian',
                 'random_params': (None, None)}
# random initial-values in limited-value-space (for motor with one state)
uniform_init = {'random_init': 'uniform',
                'interval': [[20, 40]]}
# initial value for PolynomialLoad
# (other loads can recieve an own intial value, like in external_speed_profile.py)
load_init = {'states': {'omega': 20}}


if __name__ == '__main__':

    env = gem.make(
            # choose motor
            'DcSeriesCont-v1',
            #'PMSMCont-v1',

            motor_initializer=gaussian_init,
            load_initializer=load_init,
            #visualization=MotorDashboard(plots=['omega','i_sq', 'i_sd', 'reward'],
            #                             dark_mode=True),
            visualization=MotorDashboard(plots=['omega','i', 'reward'],
                                         dark_mode=True),
            motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),
            # Take standard class and pass parameters (Load)
            load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.06),
            # Pass a string (with extra parameters)
            ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
            # Pass a Class with extra parameters
            reference_generator=rg.SwitchedReferenceGenerator(
                sub_generators=[
                    rg.SinusoidalReferenceGenerator, rg.WienerProcessReferenceGenerator(), rg.StepReferenceGenerator()
                ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
            )
        )

    # use correct controller for used motor
    #controller = Controller.make('pmsm_p_controller', env)
    controller = Controller.make('pi_controller', env)
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in range(50000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)

        if done:
            env.reset()
        cum_rew += reward
    print(cum_rew)




