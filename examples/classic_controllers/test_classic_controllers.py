from classic_controllers import Controller
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import StepReferenceGenerator, SinusoidalReferenceGenerator, SwitchedReferenceGenerator, MultipleReferenceGenerator, SubepisodedReferenceGenerator, WienerProcessReferenceGenerator, TriangularReferenceGenerator, ConstReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

ref = 'omega'
    env = gem.make(
        #'DcExtExDisc-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e'], visu_period=1), motor_parameter=dict(j_rotor=0.00005), load_parameter={'a': 0, 'b': 0, 'c': 0, 'j_load': 0},
        #'DcPermExDisc-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcSeriesDisc-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcShuntDisc-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i_a', 'i_e', 'u'], visu_period=1), motor_parameter=dict(j_rotor=0.001), load_parameter={'a': 0, 'b': 0, 'c': 0, 'j_load': 0.001},

        #'DcExtExCont-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e'], visu_period=1), motor_parameter=dict(j_rotor=0.00005), load_parameter={'a': 0, 'b': 0, 'c': 0, 'j_load': 0},
        'DcPermExCont-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcSeriesCont-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcShuntCont-v1', visualization=MotorDashboard(plots=['omega', 'torque', 'i_a', 'i_e', 'u'], visu_period=1), motor_parameter=dict(j_rotor=0.001), load_parameter={'a': 0, 'b': 0, 'c': 0.5, 'j_load': 0},

        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        #load=ConstantSpeedLoad(omega_fixed=50 * np.pi / 30),
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator(reference_state=ref, amplitude_range=(0, 0.3), offset_range=(0, 0.2)), rg.WienerProcessReferenceGenerator(reference_state=ref, amplitude_range=(0, 0.3), offset_range=(0, 0.2)), rg.StepReferenceGenerator(reference_state=ref, amplitude_range=(0, 0.3), offset_range=(0, 0.2))
            ], p=[0.5, 0.25, 0.25], super_episode_length=(10000, 100000)
        ),


    )


    try:
        controller = Controller.make('pi_controller', env, p_gain= 5, param_dict={'p_gain': 10, 'i_gain': 15})
        steps = 10000
    except:
        controller = Controller.make('three_point', env, param_dict={})
        steps = 100000

    controller = Controller.make('cascaded_controller', env,
                                  inner_controller={'controller_type': 'pi_controller'},
                                  outer_controller={'controller_type': 'pi_controller', 'p_gain': 10, 'i_gain': 15})
    '''

    controller = Controller.make('cascaded_controller', env,
                                      inner_controller={'controller_type': 'three_point'},
                                      outer_controller={'controller_type': 'on_off'})

    
    controller = Controller.make('cascaded_controller', env,
                                  inner_controller={'controller_type': 'pi_controller'},
                                  outer_controller={'controller_type': 'three_point'})

    
    
    controller = Controller.make('cascaded_controller', env,
                                  outer_controller={'controller_type': 'pid_controller'},
                                  inner_controller={'controller_type': 'on_off'})
    '''


    state, reference = env.reset()
    cum_rew = 0

    for i in range(steps):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
            controller.reset()
        cum_rew += reward
    print(cum_rew)
    env.close()
    plt.show(block=True)
