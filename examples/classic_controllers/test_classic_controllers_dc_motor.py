from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from matplotlib import pyplot as plt
import matplotlib

if __name__ == '__main__':

    """
    motor type:     'PermExDc'  Permanently Excited DC Motor
                    'ExtExDc'   Externally Excited MC Motor
                    'SeriesDc'  DC Series Motor
                    'ShuntDc'   DC Shunt Motor
                    
    control type:   'S'         Speed Control
                    'T'         Torque Control
                    'C'         Current Control
                    
    modelling:      'Cont'      Continuous Action Space
                    'Finite'    Discrete Action Space
    """

    motor_type = 'SeriesDc'
    control_type = 'T'
    modelling = 'Cont'

    motor = modelling + '-' + control_type + 'C-' + motor_type + '-v0'

    if motor_type in ['PermExDc', 'SeriesDc']:
        states = ['omega', 'torque', 'i', 'u']
    elif motor_type == 'ShuntDc':
        states = ['omega', 'torque', 'i_a', 'i_e', 'u']
    elif motor_type == 'ExtExDc':
        states = ['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e']

    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in states]

    matplotlib.use('TkAgg')
    env = gem.make(motor, visualization=MotorDashboard(additional_plots=external_ref_plots))

    controller = Controller.make(env, external_ref_plots=external_ref_plots)
    steps = 10001
    state, reference = env.reset()

    for i in range(steps):
        action = controller.control(state, reference)
        env.render()
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
            controller.reset()
    env.close()
    plt.show(block=True)
