from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':

    ref_states = ['i_sd', 'i_sq']

    q_generator = rg.SwitchedReferenceGenerator(
        sub_generators=[
            rg.TriangularReferenceGenerator(reference_state=ref_states[1], amplitude_range=(0, 0.5), offset_range=(0, 0.3)),
            rg.WienerProcessReferenceGenerator(reference_state=ref_states[1]),
            rg.StepReferenceGenerator(reference_state=ref_states[1], amplitude_range=(0, 0.5), offset_range=(0, 0.3)),
            rg.SinusoidalReferenceGenerator(reference_state=ref_states[1], amplitude_range=(0, 0.5), offset_range=(0, 0.3))],
        p=[0.3, 0.2, 0.3, 0.2], super_episode_length=(1000, 10000)
    )


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
    '''
    reference_generator = rg.MultipleReferenceGenerator([d_generator, q_generator])
    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in ['omega', 'torque', 'i_sd', 'i_sq', 'u_sd', 'u_sq']]
    matplotlib.use('TkAgg')
    env = gem.make(

        #'AbcCont-TC-SynRM-v0', visualization=MotorDashboard(additional_plots=external_ref_plots),
        'AbcCont-CC-PMSM-v0', visualization=MotorDashboard(additional_plots=external_ref_plots),

        ode_solver='scipy.solve_ivp',

        reference_generator=reference_generator,
    )

    controller = Controller.make(env, external_ref_plots=external_ref_plots)
    state, reference = env.reset()

    steps = 10001
    cum_rew = 0

    for i in range(steps):

        action = controller.control(state, reference)
        env.render()
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
            controller.reset()
        cum_rew += reward
    print(cum_rew)
    env.close()

    plt.show(block=True)
