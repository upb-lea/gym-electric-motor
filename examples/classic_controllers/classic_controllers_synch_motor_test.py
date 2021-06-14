from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad, ExternalSpeedLoad
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':

    ref_states = ['omega']
    reference_generator = rg.SinusoidalReferenceGenerator(reference_state=ref_states[0], frequency_range=(0.4, 0.5), offset_range=(0.7, 0.75), amplitude_range=(0.2, 0.3), episode_lengths=30000)

    
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq']]
    matplotlib.use('qt5agg')

    env = gem.make(
        #'Finite-TC-SynRM-v0', visualization=MotorDashboard(additional_plots=external_ref_plots),
        #'Finite-TC-PMSM-v0', visualization=MotorDashboard(additional_plots=external_ref_plots),

        #'AbcCont-TC-SynRM-v0', visualization=MotorDashboard(additional_plots=external_ref_plots),
        'AbcCont-TC-PMSM-v0', visualization=MotorDashboard(additional_plots=external_ref_plots, update_interval=1000),

        ode_solver='scipy.solve_ivp',
        motor=dict(nominal_values=dict(omega=10e3 * np.pi / 30, torque=95.0, i=240, epsilon=np.pi, u=300),
                   motor_parameter=dict(p=3, l_d=0.37e-3, l_q=1.2e-3, j_rotor=0.03883, r_s=18e-3, psi_p=45e-3),
                   limit_values=dict(omega=12e3 * np.pi / 30, torque=100, i=280, u=320)),
        #load=ConstantSpeedLoad(7.8e3 * np.pi / 30),
        #load=ExternalSpeedLoad(lambda t: ((np.sin(2 * np.pi * 5 * t) + 1) * 2e3 + 5e3) * np.pi / 30),
        load=PolynomialStaticLoad(),
        reference_generator=reference_generator,
    )

    controller = Controller.make(env, external_ref_plots=external_ref_plots, torque_control='online')
    state, reference = env.reset()
    steps = 10001
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
