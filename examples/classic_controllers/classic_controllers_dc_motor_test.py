from classic_controllers import Controller

import gym_electric_motor as gem
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad, ConstantSpeedLoad
from gym.spaces import Box
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor import reference_generators as rg
from matplotlib import pyplot as plt
import numpy as np
import matplotlib



if __name__ == '__main__':

    ref = ['omega']

    if len(ref) == 1:
        reference_generator = rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator(reference_state=ref[0], amplitude_range=(0, 0.3), offset_range=(0, 0.2)),
                rg.WienerProcessReferenceGenerator(reference_state=ref[0], amplitude_range=(0, 0.3),
                                                   offset_range=(0, 0.2)),
                rg.StepReferenceGenerator(reference_state=ref[0], amplitude_range=(0, 0.3), offset_range=(0, 0.2))
            ], p=[0.5, 0.25, 0.25], super_episode_length=(10000, 100000)
        )
    else:
        ref_gen = []
        for r in ref:
            ref_gen.append(rg.SwitchedReferenceGenerator(
                    sub_generators=[
                    rg.SinusoidalReferenceGenerator(reference_state=r, amplitude_range=(0, 0.3), offset_range=(0, 0.2)),
                    rg.WienerProcessReferenceGenerator(reference_state=r, amplitude_range=(0, 0.3),
                                                       offset_range=(0, 0.2)),
                    rg.StepReferenceGenerator(reference_state=r, amplitude_range=(0, 0.3), offset_range=(0, 0.2))
                    ], p=[0.5, 0.25, 0.25], super_episode_length=(10000, 100000)
                )
            )
        reference_generator = rg.MultipleReferenceGenerator(ref_gen)

    limit_values = dict(omega=100*np.pi/30)
    matplotlib.use('TkAgg')
    env = gem.make(
        #'DcExtExDisc-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e'], visu_period=1),
        #'DcPermExDisc-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcSeriesDisc-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcShuntDisc-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_a', 'i_e', 'u'], visu_period=1),

        #'DcExtExCont-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e'], visu_period=1),
        'DcPermExCont-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcSeriesCont-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i', 'u'], visu_period=1),
        #'DcShuntCont-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_a', 'i_e', 'u'], visu_period=1),

        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        #limit_values=limit_values,
        #load=ConstantSpeedLoad(omega_fixed=50 * np.pi / 30),

        reference_generator=reference_generator
    )
    controller = Controller.make(env)

    steps = 10001 if type(env.action_space) == Box else 100001
    state, reference = env.reset()
    cum_rew = 0

    for i in range(steps):

        action = controller.control(state, reference)
        env.render(ref=controller.plot())
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
            controller.reset()
        cum_rew += reward
    print(cum_rew)
    env.close()
    plt.show(block=True)
