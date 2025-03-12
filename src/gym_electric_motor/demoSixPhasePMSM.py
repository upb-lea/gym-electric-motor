import gym_electric_motor as gem
from gym_electric_motor.reference_generators import LaplaceProcessReferenceGenerator

 # Select a different ode_solver with default parameters by passing a keystring
env = gem.make(
         "Cont-CC-SIXPMSM-v0",)
terminated = True
for _ in range(1000):
   if terminated:
     state, reference = env.reset()
   (state, reference), reward, terminated, truncated, _ = env.step(env.action_space.sample())