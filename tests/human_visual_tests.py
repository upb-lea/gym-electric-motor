from gym_electric_motor.envs.gym_pmsm import *
import numpy as np


"""
tun this script to see if the visualization looks as expected
"""
env = ContPermanentMagnetSynchronousMotorEnvironment(visualization='MotorDashboard',
                                                     plotted_variables=['omega', 'torque', 'u_a', 'u_b', 'u_c'],
                                                     visu_period=1,
                                                     update_period=3E-2)
tau = 1E-4
env.reset()
for k in range(25000):
    action = 0.3 * np.array(
        [np.sin(k * tau / 2E-2), np.sin(k * tau / 2E-2 + np.pi / 3), np.sin(k * tau / 2E-2 - np.pi / 3)])
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()

env.close()
