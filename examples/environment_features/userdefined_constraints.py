import time
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append('..')
from classic_controllers.simple_controllers import Controller
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad

'''
This code example presents how the constraint monitor can be used to define custom termination conditions.
This feature is interesting for e.g. three-phase drives, where the current needs to be monitored with
respect to the total current within all three phases , and not separately per phase.
In this example we will define an external constraint for the DcSeriesMotor which lies at 95 % of
the predefined current limit.

For a more general introduction to GEM, we recommend to have a look at the "_control.py" examples first.
'''


# the following external monitors are examples for the general use-case.
class ExternalMonitorClass:
    """
    A Class for defining a ConstraintMonitor. The general structure of the
    class is arbitrary, but a __call__() method, with a return-value out of
    [0, 1] is necessary. The return indicates the violation of the
    constraint-conditions
    """
    def __init__(self):
        # Do some setup
        self.upper_current_constraint = 95
        self.lower_current_constraint = -95

    def __call__(self, state, observed_states, **kwargs):

        # Read the current limit and the time since the last reset
        physical_system = kwargs['physical_system']
        current_limit = physical_system._limits[2]
        time = kwargs['k'] * physical_system.tau

        # the result of this check sets the "done" flag of the env.step() output
        # done==True usually necessitates an env.reset()
        return self.check_violation(denormalized_current=state[2] * current_limit, t=time)

    def check_violation(self, denormalized_current, t):

        # check whether the constraints are violated
        check_violation = int(abs(denormalized_current) < self.lower_current_constraint or
                              abs(denormalized_current) > self.upper_current_constraint)

        if check_violation:
            print(f"External current constraint was violated at t={t} s")
            print(f"Measured current: {denormalized_current} A, constraint was at {self.upper_current_constraint} A")
            print()

        return check_violation


def external_monitor_func(state, physical_system, k, **kwargs):
    """
    external monitor function with a return-value out of [0, 1], indicating the
    violation of the constraint conditions. Additional function parameters
    can be given to the ConstraintMonitor separately
    """
    # get the defined current limit
    current_limit = physical_system._limits[2]

    # denormalize the current value
    denormalized_current = state[2] * current_limit

    # Define an extra constraint at 95 % of the current
    extra_constraint = 0.95 * current_limit

    # check whether the constraint is violated
    check_violation = int(abs(denormalized_current) > extra_constraint)

    if check_violation:
        print(f"External current constraint was violated at t={k * physical_system.tau} s")
        print(f"Measured current: {denormalized_current} A, constraint was at {extra_constraint} A")
        print()

    # the result of this check sets the "done" flag of the env.step() output
    # done==True usually necessitates an env.reset()
    return check_violation


if __name__ == '__main__':
    env = gem.make(
        'DcSeriesCont-v1',
        visualization=MotorDashboard(state_plots=['omega', 'i']),
        load=ConstantSpeedLoad(omega_fixed=10),
        ode_solver='scipy.solve_ivp',

        # Here, we set a reference that is within the usual current limits (99 % of the limit value)
        reference_generator=rg.ConstReferenceGenerator(reference_state='i', reference_value=0.99),

        # However, we now use the external constraint monitor to set an external constraint of 95 % of the current limit
        # So now, the reference at 99 % of the current limit would lead the drive to
        # violate our externally defined constraints
        # If we comment out the constraint monitor, the simulation should run without any violation
        constraint_monitor=ExternalMonitorClass(),
        #constraint_monitor=external_monitor_func,
    )

    # After setup, start the simulation
    controller = Controller.make('pi_controller', env)
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in range(10000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)

        if done:
            # Reset the env and the controller after termination
            state, reference = env.reset()
            controller.reset()
        cum_rew += reward
    env.close()

    print(cum_rew)
