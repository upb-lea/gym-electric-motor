from examples.agents.simple_controllers import Controller
import time
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard

"""
Run this file from within the 'examples' folder:
>> cd examples
>> python perm_dc_omega.py

Description:
        Environment to control a Continuously controlled DC Permanently Excited Motor.

        Controlled Quantity: 'omega'

        Limitations: Physical limitations of the motor like Current,Speed

        Converter : FourQuadrantConverter from converters.py

"""

if __name__ == '__main__':

    """Discrete mode: It is the mode where the input command causes the controlled variable to change its 
    value in discrete steps,with a brief static condition occurring between each step.
    
    Continuous mode is the one where the input command signal causes  the controlled variable to remain constant,
    or to change in a linear(rather than step) mode. 
    """
    env = gem.make(
        'DcPermExCont-v1',  # replace with 'DcPermExDisc-v1' for continuous mode
        visualization=MotorDashboard(plots=['omega', 'torque', 'i', 'u', 'u_sup'], visu_period=1),
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator, rg.WienerProcessReferenceGenerator(), rg.StepReferenceGenerator()
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )
    )
    controller = Controller.make('p_controller', env)
    # The above controller can be replaced with 'on_off'(disc),three_point(disc),'p_controller', 'cascaded pi' from simple_controllers class
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in range(100000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
        cum_rew += reward
    print(cum_rew)
