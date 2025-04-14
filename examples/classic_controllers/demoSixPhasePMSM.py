import gym_electric_motor as gem
from gym_electric_motor.reference_generators import LaplaceProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.render_modes import RenderMode
from gym_electric_motor.visualization.console_printer import ConsolePrinter
from gym_electric_motor.physical_systems import ConstantSpeedLoad


my_changed_voltage_supply_args = {'u_nominal': 200.0}
motor_dashboard = MotorDashboard(state_plots=("i_a1","i_a2"),render_mode=RenderMode.Figure)
load = ConstantSpeedLoad(omega_fixed=200)
env = gem.make(
         "Cont-CC-SIXPMSM-v0",
         supply = my_changed_voltage_supply_args,
         load = load,
         visualization = motor_dashboard)
terminated = True
for _ in range(1000):
   if terminated:
     state, reference = env.reset()
   (state, reference), reward, terminated, truncated, _ = env.step(0 * env.action_space.sample())

motor_dashboard.initialize()
fig = motor_dashboard.figure()
fig.gca().set_ylim([-20,+20])
fig.gca().set_xlim([0,0.02])
fig.get_axes()[0].set_ylim([-20,+20])
fig.gca().legend(loc='upper right')
fig.get_axes()[0].legend(loc='upper right')
fig.savefig("test")
motor_dashboard.save_to_file()


#console.render()
env.close()