## Examples

In order to provide better accessibility of this simulation library we created several examples that showcase the usage of GEM's interface and features.
The presented examples can be classified either as classical control examples (where conventional control methods are applied), as reinforcement-learning control examples 
(which present a rather novel branch of control theory), or as feature showcases (where the focus lies in introducing interesting and useful builtins).

### Classical Control
- [pi_series_omega_control.py](classic_controllers/pi_series_omega_control.py): a conventional linear control algorithm applied to the speed control of a series DC motor with continuous-control-set
- [perm_dc_omega.py](classic_controllers/perm_dc_omega.py): a conventional discontinuous control algorithm applied to the speed control of a permanently excited DC motor with finite-control-set
- [pmsm_foc.py](classic_controllers/pmsm_foc.py): a field-oriented control example for three-phase permanent magnet synchonous motors (PMSM) with standard PI-controllers (continuous-control-set).

### Reinforcement-Learning Control
- [dqn_series_current_control.py](reinforcement_learning_controllers/dqn_series_current_control.py): a deep Q-value network reinforcement-learning control approach for finite-control-set current control of a series DC motor
- [ddpg_pmsm_dq_current_control.py](reinforcement_learning_controllers/ddpg_pmsm_dq_current_control.py): a deep deterministic policy gradient reinforcement-learning control approach applied to the current control of a permanent magnet synchronous motor within the $dq$-frame with continuous-control-set
- [ddpg_series_omega_control.py](reinforcement_learning_controllers/ddpg_series_omega_control.py): a deep deterministic policy gradient reinforcement-learning control approach applied to the speed control of a series DC motor with continuous-control-set

### Feature Showcases
- [external_speed_profile.py](environment_features/external_speed_profile.py): presents a builtin feature that can be used to define arbitrary speed profiles, which is useful when e.g. investigating generator operation (where mechanical and thus electrical frequency is determined by external means) 
- [userdefined_initialization.py](environment_features/userdefined_initialization.py): presents a builtin feature that allows the user to determine the initial state of the motor, which comes in handy when e.g. using exploring starts in reinforcement learning applications
- [userdefined_constraints.py](environment_features/userdefined_constraints.py): presents a builtin feature that can be used to extend the operational constraints of the motor, because oftentimes emergency shutdowns should be performed on the basis of composite conditions (e.g. for three-phase drives it is important to monitor <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;i_s=\sqrt{i_d^2&plus;i_q^2}" title="i_s=\sqrt{i_d^2+i_q^2}" /> instead of monitoring <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;i_d" title="i_d" /> and <img src="https://latex.codecogs.com/png.latex?\inline&space;\bg_white&space;i_q" title="i_q" /> independently)
- [scim_ideal_grid_simulation.py](environment_features/scim_ideal_grod_simulation.py): simulates the start-up behavior of the squirrel cage induction motor connected to an ideal three-phase grid. 
The state and action space is continuous.
Running the example will create a formatted plot that show the motors angular velocity, the drive torque, the applied voltage in three-phase abc-coordinates and the measured current in field-oriented dq-coordinates.