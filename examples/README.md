## Examples

In order to provide better accessibility of this simulation library we created several examples that showcase the usage of GEM's interface and features.
The presented examples can be classified either as classical control approaches, as optimal control using model predictive control and model-free reinforcement-learning control examples, or as feature showcases (where the focus lies in introducing interesting and useful builtins).

### Installation, Setup and Interface
- [GEM_cookbook.ipynb](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/environment_features/GEM_cookbook.ipynb): a basic tutorial-style notebook that presents the basic interface and usage of GEM

### Classical Control
- [Classic controller suite](classic_controllers/classic_controllers.py): a control algorithm library using standard, expert-driven techniques with automatic and manual controller tuning for an increasingly growing number of motor and inverter combinations
- [Classic controller examples](classic_controllers): a set of examples applying the classic controller suite (e.g., field-oriented control for permanent synchronous motors)

### Advanced Control
- [gekko_mpc_cont_pmsm_example.ipynb](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/model_predictive_controllers/gekko_mpc_cont_pmsm_example.ipynb): a model predictive control solution for the currents of the three-phase permanent magnet synchronous motor on a continuous-control-set

### Reinforcement-Learning Control
- [dqn_series_current_control.py](reinforcement_learning_controllers/dqn_series_current_control.py): a deep Q-value network reinforcement-learning control approach for finite-control-set current control of a series DC motor
- [ddpg_pmsm_dq_current_control.py](reinforcement_learning_controllers/ddpg_pmsm_dq_current_control.py): a deep deterministic policy gradient reinforcement-learning control approach applied to the current control of a permanent magnet synchronous motor within the $dq$-frame with continuous-control-set
- [ddpg_series_omega_control.py](reinforcement_learning_controllers/ddpg_series_omega_control.py): a deep deterministic policy gradient reinforcement-learning control approach applied to the speed control of a series DC motor with continuous-control-set
- [keras_rl2_dqn_disc_pmsm_example.ipynb](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/keras_rl2_dqn_disc_pmsm_example.ipynb): a tutorial-style notebook that presents the usage of GEM in conjunction with [Keras_RL2](https://github.com/wau/keras-rl2) in the context of deep Q learning current control of a permanent magnet synchronous motor
- [stable_baselines3_dqn_disc_pmsm_example.ipynb](https://colab.research.google.com/github/upb-lea/gym-electric-motor/blob/master/examples/reinforcement_learning_controllers/stable_baselines3_dqn_disc_pmsm_example.ipynb): a tutorial-style notebook that presents the usage of GEM in conjunction with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) in the context of deep Q learning current control of a permanent magnet synchronous motor

### Feature Showcases
- [external_speed_profile.py](environment_features/external_speed_profile.py): presents a builtin feature that can be used to define arbitrary speed profiles, which is useful when e.g. investigating generator operation (where mechanical and thus electrical frequency is determined by external means) 
- [userdefined_initialization.py](environment_features/userdefined_initialization.py): presents a builtin feature that allows the user to determine the initial state of the motor, which comes in handy when e.g. using exploring starts in reinforcement learning applications
- [scim_ideal_grid_simulation.py](environment_features/scim_ideal_grid_simulation.py): simulates the start-up behavior of the squirrel cage induction motor connected to an ideal three-phase grid. 
The state and action space is continuous.
Running the example will create a formatted plot that show the motors angular velocity, the drive torque, the applied voltage in three-phase abc-coordinates and the measured current in field-oriented dq-coordinates.