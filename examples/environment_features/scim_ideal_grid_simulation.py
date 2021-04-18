"""Run this file from within the 'examples' folder:
>> cd examples
>> python dqn_series_current_control.py
"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
import matplotlib.pyplot as plt

# This example simulates the start-up behavior of the squirrel cage induction motor connected to
# an ideal three-phase grid. The state and action space is continuous.
# Running the example will create a formatted plot that show the motors angular velocity, the drive torque,
# the applied voltage in three-phase abc-coordinates and the measured current in field-oriented dq-coordinates.


def parameterize_three_phase_grid(amplitude, frequency, initial_phase):
    """This nested function allows to create a function of time, which returns the momentary voltage of the
     three-phase grid.

    The nested structure allows to parameterize the three-phase grid by amplitude(as a fraction of the DC-link voltage),
    frequency (in Hertz) and initial phase (in degree).
    """

    omega = frequency * 2 * np.pi  # 1/s
    phi = 2 * np.pi / 3  # phase offset
    phi_initial = initial_phase * 2 * np.pi / 360

    def grid_voltage(t):
        u_abc = [
            amplitude * np.sin(omega * t + phi_initial),
            amplitude * np.sin(omega * t + phi_initial - phi),
            amplitude * np.sin(omega * t + phi_initial + phi)
         ]
        return u_abc
    return grid_voltage


# Create the environment
env = gem.make(
    # Choose the squirrel cage induction motor (SCIM) with continuous-control-set
    "AbcCont-CC-SCIM-v0",

    # Define the numerical solver for the simulation
    ode_solver="scipy.ode",

    # Define which state variables are to be monitored concerning limit violations
    # "()" means, that limit violation will not necessitate an env.reset()
    constraints=(),

    # Parameterize the mechanical load (we set everything to zero such that only j_rotor is significant)
    load=dict(load_parameter=(dict(a=0, b=0, c=0, j_load=0))),

    # Set the sampling time
    tau=1e-5
)

tau = env.physical_system.tau
limits = env.physical_system.limits

# reset the environment such that the simulation can be started
(state, reference) = env.reset()

# We define these arrays in order to save our simulation results in them
# Initial state and initial time are directly inserted
STATE = np.transpose(np.array([state * limits]))
TIME = np.array([0])

# Use the previously defined function to parameterize a three-phase grid with an amplitude of
# 80 % of the DC-link voltage and a frequency of 50 Hertz
f_grid = 50  # Hertz
u_abc = parameterize_three_phase_grid(amplitude=0.8, frequency=f_grid, initial_phase=0)

# Set a time horizon to simulate, in this case 60 ms
time_horizon = 0.06
step_horizon = int(time_horizon / tau)
for idx in range(step_horizon):
    # calculate the time of this simulation step
    time = idx * tau

    # apply the voltage as given by the grid
    (state, reference), reward, done, _ = env.step(u_abc(time))

    # save the results of this simulation step
    STATE = np.append(STATE, np.transpose([state * limits]), axis=1)
    TIME = np.append(TIME, time)

# convert the timescale from s to ms
TIME *= 1e3

# the rest of the code is for plotting the results in a nice way
# the state indices for the SCIM are:
# STATE[0]: omega (mechanical angular velocity)
# STATE[1]: T (drive torque)
# STATE[2] - STATE[4]: i_sa, i_sb, i_sc (three-phase stator currents)
# STATE[5] - STATE[6]: i_sd, i_sq (stator currents in field oriented dq-coordinates)
# STATE[7] - STATE[9]: u_sa, u_sb, u_sc (three-phase stator voltages)
# STATE[10] - STATE[11]: u_sd, u_sq (stator voltages in field oriented dq-coordinates)
# STATE[12]: epsilon (rotor angular position)
# STATE[13]: u_sup (DC-link supply voltage)

plt.subplots(2, 2, figsize=(7.45, 2.5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.08, hspace=0.05)
plt.rcParams.update({'font.size': 8})

plt.subplot(2, 2, 1)
plt.plot(TIME, STATE[0])
plt.ylabel(r"$\omega_\mathrm{me} \, / \, \frac{1}{\mathrm{s}}$")
plt.xlim([0, 60])
plt.yticks([0, 50, 100, 150])
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.tick_params(axis='both', direction="in", left=True, right=False, bottom=True, top=True)
plt.grid()

ax = plt.subplot(2, 2, 2)
plt.plot(TIME, STATE[7], label=r"$u_a$")
plt.plot(TIME, STATE[8], label=r"$u_b$")
plt.plot(TIME, STATE[9], label=r"$u_c$")
plt.ylabel(r"$u \, / \, \mathrm{V}$")
plt.xlim([0, 60])
plt.yticks([-200, 0, 200])
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tick_params(axis='x', which='both', labelbottom=False)
plt.tick_params(axis='both', direction="in", left=False, right=True, bottom=True, top=True)
plt.grid()
plt.legend(loc="lower right", ncol=3)

plt.subplot(2, 2, 3)
plt.plot(TIME, STATE[1])
plt.xlabel(r"$t \, / \, \mathrm{ms}$")
plt.ylabel(r"$T \, / \, \mathrm{Nm}$")
plt.xlim([0, 60])
plt.yticks([0, 20])
plt.tick_params(axis='both', direction="in", left=True, right=False, bottom=True, top=True)
plt.grid()

ax = plt.subplot(2, 2, 4)
plt.plot(TIME, STATE[5], label=r"$i_d$")
plt.plot(TIME, STATE[6], label=r"$i_q$")
plt.xlabel(r"$t \, / \, \mathrm{ms}$")
plt.ylabel(r"$i \, / \, \mathrm{A}$")
plt.xlim([0, 60])
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
plt.tick_params(axis='both', direction="in", left=False, right=True, bottom=True, top=True)
plt.yticks([0, 10, 20, 30])
plt.grid()
plt.legend(loc="upper right", ncol=2)

plt.show()
