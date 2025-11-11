from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import gym_electric_motor as gem
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.physical_systems.mechanical_loads.external_speed_load import ExternalSpeedLoad
from gym_electric_motor.physical_systems.solvers import EulerSolver
from gym_electric_motor.reference_generators import LaplaceProcessReferenceGenerator
from gym_electric_motor.reference_generators.const_reference_generator import ConstReferenceGenerator
from gym_electric_motor.reward_functions.weighted_sum_of_errors import WeightedSumOfErrors
from gym_electric_motor.visualization import MotorDashboard, motor_dashboard
from gym_electric_motor.visualization.motor_dashboard_plots import action_plot
from gym_electric_motor.visualization.render_modes import RenderMode
from gym_electric_motor.visualization.console_printer import ConsolePrinter
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gymnasium.spaces import Discrete
from gym_electric_motor.visualization import MotorDashboard

# define motor arguments
torque_ref_generator = ConstReferenceGenerator(reference_state='torque', reference_value=np.random.uniform(-1, 1))

motor_parameter = dict(p=3,            # [p] = 1, nb of pole pairs
                       r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                       l_d=0.37e-3,    # [l_d] = H, d-axis inductance
                       l_q=1.2e-3,     # [l_q] = H, q-axis inductance
                       psi_p=65.65e-3, # [psi_p] = Vs, magnetic flux of the permanent magnet
                       )  # BRUSA

u_sup = 350
nominal_values=dict(omega=12000*2*np.pi/60,
                    i=240,
                    u=u_sup)

limit_values=nominal_values.copy()
limit_values["i"] = 270
limit_values["torque"] = 200

sampling_time = 50e-6

visualization = MotorDashboard(state_plots=['i_sq', 'i_sd'], reward_plot=True) 

env = gem.make("Finite-TC-PMSM-v0",
                   motor = dict(
                       motor_parameter=motor_parameter,
                       limit_values=limit_values,
                       nominal_values=nominal_values,
                   ),
                   supply=dict(u_nominal=u_sup),
                   load = ConstantSpeedLoad(omega_fixed=200),
                   tau=sampling_time,
                   reward_function=WeightedSumOfErrors(reward_weights={'torque': 1},  # but the reward distribution will be overwritten
                                                              gamma=0.868), # by means of the defined wrapper function
                   reference_generator=torque_ref_generator,
                   visualization = visualization
                   )
terminated = True
#visualization.initialize()


varying_T_s_Array = [25e-6, 50e-6]#calculation verified for 10 and 50 as well
current_index = [0]

def T_s_selector(array):
    value = array[current_index[0]]
    return value

#for plotting
i_sd_idx = env.get_wrapper_attr('state_names').index('i_sd')
i_sq_idx = env.get_wrapper_attr('state_names').index('i_sq')
i_sdfirstEpisode = []
i_sdSecondEpisode = []
timepointsfirstEpisode = []
timepointsSecondEpisode = []
i_sqfirstEpisode = []
i_sqSecondEpisode = []
i_sd = []
i_sq = []
resetCounter = 0

for _ in range(1000):
   if terminated:
     state, reference = env.reset(options={"varying_T_s": T_s_selector(varying_T_s_Array)})
     resetCounter = resetCounter + 1
     #state, reference = env.reset()
   #env.render()
   current_index[0] = 1 - current_index[0]
   (state, reference), reward, terminated, truncated, _ = env.step(0* env.action_space.sample())
   if (resetCounter == 2):
      timepointsSecondEpisode.append(env.env.env._physical_system.k * env.env.env._physical_system._tau)
      i_sdSecondEpisode.append(state[i_sd_idx])
   elif(resetCounter == 1):
      timepointsfirstEpisode.append(env.env.env._physical_system.k * env.env.env._physical_system._tau)
      i_sdfirstEpisode.append(state[i_sd_idx])
      i_sqfirstEpisode.append(state[i_sq_idx])
   else:
      pass
'''
i_sd.append(state[i_sd_idx])
i_sq.append(state[i_sq_idx])
'''

addPreviousEndTime =  np.array(timepointsSecondEpisode) + timepointsfirstEpisode[-1]
fulltime = np.concatenate((timepointsfirstEpisode, addPreviousEndTime))
combined_i_d =  np.concatenate((i_sdfirstEpisode, i_sdSecondEpisode))
plt.step(fulltime,combined_i_d, where = 'post', linewidth = 2)



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex= True)

#plot for i_sd
ax1.step(timepointsfirstEpisode, i_sdfirstEpisode, label='i_sd')
ax1.set_ylabel('Normalized Values')
ax1.set_xlabel('timepointsfirstEpisode')
ax1.legend()
ax1.set_ylim([-1, 1])

#plot for i_sq
ax2.step(timepointsSecondEpisode, i_sdSecondEpisode, label='i_sd')
ax2.set_ylabel('Normalized Values')
ax2.set_xlabel('timepointsSecondEpisode')
ax2.legend()
ax2.set_ylim([-1, 1])

plt.tight_layout()
plt.show()

'''
time_points = range(len(i_sd))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

#plot for i_sd
ax1.plot(time_points, i_sd, label='i_sd', color='blue')
ax1.set_ylabel('Normalized Values')
ax1.set_xlabel('No of steps')
ax1.legend()
ax1.set_ylim([-1, 1])

#plot for i_sq
ax2.plot(time_points, i_sq, label='i_sq', color='red')
ax2.set_ylabel('Normalized Values')
ax2.set_xlabel('No of steps')
ax2.legend()
ax2.set_ylim([-1, 1])

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()
'''