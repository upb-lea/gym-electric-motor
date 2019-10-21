# Gym Electric Motor

The gym-electric-motor package is a software toolbox for the simulation of different electric motors. 
The toolbox is built upon the [OpenAI Gym Environments](https://gym.openai.com/) for reinforcement learning. 
Therefore, the toolbox is especially designed for running reinforcement learning algorithms to train agents to control electric motors.
So far, several DC-motor models and a permanent magnet synchronous motor (PMSM) are available. Beside electrical 
motors, also converters and load models are implemented. The converters can be driven by means of a duty cycle (continuous mode) or 
switching commands (discrete mode). 
The figure shows the basic scheme of the converter, motor and load. 
### Physical Structure of the Environments Components
![](docs/plots/FigureConvMotorLoad6.svg)
### Control Flow of a Step Cycle of the Environment 
![](docs/plots/CycleScheme.svg)

## Installing

- Install gym-electric-motor from PyPI (recommended):

```
pip install gym-electric-motor
```

- Install from Github source:

```
git clone git@github.com:upb-lea/gym-electric-motor.git
cd gym-electric-motor
# Then either
python setup.py install
# or alternatively
pip install -e .
```
## Authors
Arne Traue, Gerrit Book

## Important notes
Latest readme update: 01-10-2019


## Getting started
Like every gym environment, the basic user interface consists of four main functions.

* `gym.make(environment-id, **kwargs)`  
    Returns an instantiated motor environment. Call this function at the beginning.
 
* `env.reset()`  
    Resets the motor. This includes a new initial state and new reference trajectories.
    Call this function before a new episode starts. 

* `env.step(action)`      
    Simulate one single time step on the motor with an action.
    Call this function iteratively until termination is reached.

* `env.render()`    
    Update the visualization of the motor states.

### Gym Make Call
The make function takes the environment-ids and several constructor arguments.
With the environment-id you select a certain motor type and action type (continuous or discrete) and with the further 
constructor arguments you can parametrize the environment to your control problem.

##### Environment Ids

* `'emotor-dc-extex-cont-v0'`     
    externally excited DC motor with continuous actions.
 
* `'emotor-dc-extex-disc-v0'`   
    externally excited DC motor with discrete actions.

* `'emotor-dc-permex-cont-v0'`    
    permanently excited DC motor with continuous actions.

* `'emotor-dc-permex-disc-v0'`    
    permanently excited DC motor with discrete actions.

* `'emotor-dc-shunt-cont-v0'`    
    DC shunt motor with continuous actions.

* `'emotor-dc-shunt-disc-v0'`    
    DC shunt motor with discrete actions.

* `'emotor-dc-series-cont-v0'`    
    DC series motor with continuous actions.

* `'emotor-dc-series-disc-v0'`  
    DC series motor with discrete actions.
    
* `'emotor-pmsm-cont-v0'`:  
    permanent magnet synchronous motor with continuous actions.

* `'emotor-pmsm-disc-v0'`:  
    permanent magnet synchronous motor with discrete actions.

#### Make Arguments
With the constructor arguments you can parametrize the control problem. 
You can select, which motor model to take, which quantities to control, etc.
For further information have a look at [Link for the APIDOC Base Env Site]

### Reset
The reset function determines new references, new initial values and resets the visualization.
Call this function before a new episode begins.
The parameters of the motor, converter and load will be those during instantiation.

### Step
This function performs one action on the environment for one time step.
It simulates the motor and needs to be called in every time step. It takes the action as parameter only.
First the input voltage to the motor from the converter is determined and afterwards an integrator is used to compute 
the next state. 
Eventually, the reward is evaluated and returned together with the next observation and a flag indicating termination.
Several reward functions are available.

### Render
The visualization contains graphs of the motor quantities 'speed, voltages, currents, torque' for one episode. 
What should be shown is to be specified in the configuration-parameter.
The quantities that should be displayed can be specified in the constructor-parameters.
All visualizations are optional and recommended to be disabled for increased speed of training.

### Examples

- Conventional PI controller as speed controller on a dc series motor [(jump to source)](examples/pi_series_omega_control.py).

- Training and testing of a [Keras-rl](https://github.com/keras-rl/keras-rl) DDPG-Agent as a speed controller on a dc series motor [(jump to source)](examples/ddpg_series_omega_control.py).
 
## Motor Models
The following motor models are included:

Four DC motors:

- permanently excited motor
- externally excited motor
- series motor
- shunt motor

and the PMSM (permanent magnet synchronous motor)

### Converter
Following converters are included:

- 1 quadrant converter (1QC)

- 2 quadrant converter (2QC) as an asymmetric half bridge with both current polarities

- 4 quadrant converter (4QC)

All converters can consider interlocking times and a dead time of one sampling interval.
Furthermore, they can be controlled with a discrete action space or a continuous action space.

Discrete actions are the direct switching states of the transistors.
Continuous actions are the duty cycles for a pulse width modulation on the transistors. 

### Load
The load model consists of a quadratic load function, with user defined coefficients. 
Furthermore the moment of inertia of the load attached to the motor can be specified.

### Notes about the Parameters
All nominal values of voltages and currents are DC values in the case of a DC motor and peak phase values for the PMSM.
Therefore, data sheet values for line voltage and phase currents of a PMSM has to be transformed with:

![](docs/plots/voltagetransformation.svg)

Furthermore, the angular velocity is the mechanical one and not the electrical: 

![](docs/plots/omegame.svg)
