from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from matplotlib import pyplot as plt
import matplotlib
import numpy as np


def plot(controller, external_reference_plots, state_names):
    external_refs = controller.plot()
    external_ref_plots = list(external_reference_plots)
    ref_state_idxs = external_refs['ref_state']
    plot_state_idxs = [
        list(state_names).index(external_ref_plot.state) for external_ref_plot in external_reference_plots
    ]
    ref_values = external_refs['ref_value']
    for ref_state_idx, ref_value in zip(ref_state_idxs, ref_values):
        try:
            # Match the reference with the corresponding plot.
            plot_idx = plot_state_idxs.index(ref_state_idx)
        except ValueError:
            # Go on, if there is no fitting plot.
            pass
        else:
            external_ref_plots[plot_idx].external_reference(ref_value)


if __name__ == '__main__':

    motor_parameter = dict(r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, p=3, j_rotor=0.06)
    limit_values = dict(i=160 * 1.41, omega=12000 * np.pi / 30, u=450)
    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}

    ref_states = ['i_sd', 'torque']

    q_generator = rg.SwitchedReferenceGenerator(
        sub_generators=[
            rg.TriangularReferenceGenerator(reference_state=ref_states[1], amplitude_range=(0, 0.5), offset_range=(0, 0.3)),
            rg.WienerProcessReferenceGenerator(reference_state=ref_states[1]),
            rg.StepReferenceGenerator(reference_state=ref_states[1], amplitude_range=(0, 0.5), offset_range=(0, 0.3)),
            rg.SinusoidalReferenceGenerator(reference_state=ref_states[1], amplitude_range=(0, 0.5), offset_range=(0, 0.3))],
        p=[0.3, 0.2, 0.3, 0.2], super_episode_length=(1000, 10000)
    )

    '''
    d_generator = rg.SwitchedReferenceGenerator(
        sub_generators=[
            rg.TriangularReferenceGenerator(reference_state='i_sd', amplitude_range=(0, 0.3), offset_range=(0, 0.2)),
            rg.WienerProcessReferenceGenerator(reference_state='i_sd'),
            rg.StepReferenceGenerator(reference_state='i_sd', amplitude_range=(0, 0.3), offset_range=(0, 0.2)),
            rg.SinusoidalReferenceGenerator(reference_state='i_sd', amplitude_range=(0, 0.3), offset_range=(0, 0.2)),
            rg.ConstReferenceGenerator(reference_state='i_sd', value=0)],
        p=[0.25, 0.1, 0.25, 0.2, 0.2], super_episode_length=(1000, 10000)
    )
    '''
    d_generator = rg.ConstReferenceGenerator(reference_state=ref_states[0], reference_value=0)

    reference_generator = rg.MultipleReferenceGenerator([d_generator, q_generator])
    external_ref_plots = [ExternallyReferencedStatePlot('i_sq')]
    matplotlib.use('TkAgg')
    env = gem.make(

        #'SynRMDisc-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq'], visu_period=10),
        #'PMSMDisc-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq'], visu_period=10),

        #'SynRMCont-v1', visualization=MotorDashboard(state_plots_extended=['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq'], visu_period=10),
        'AbcCont-TC-PMSM-v0', visualization=MotorDashboard(
            state_plots=['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq'],
            additional_plots=external_ref_plots,
        ),

        ode_solver='scipy.solve_ivp',

        reference_generator=reference_generator,
    )

    controller = Controller.make(env)
    state, reference = env.reset()

    steps = 10001
    cum_rew = 0

    for i in range(steps):
        action = controller.control(state, reference)
        plot(controller, external_ref_plots, env.state_names)
        env.render()
        (state, reference), reward, done, _ = env.step(action)
        if done:
            state, reference = env.reset()
        cum_rew += reward
    print(cum_rew)
    env.close()

    plt.show(block=True)
