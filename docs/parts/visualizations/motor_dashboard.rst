Motor Dashboard
###############################

..  contents::

..  toctree::
    :maxdepth: 2
    :caption: Available Plots:

    motor_dashboard_plots

Motor Dashboard Code Docu
__________________________

.. autoclass:: gym_electric_motor.visualization.motor_dashboard.MotorDashboard
   :members:
   :inherited-members:


How To: Select plots to render
________________________________

.. code-block:: python

    import gym_electric_motor as gem
    from gym_electric_motor.visualization import MotorDashboard
    from gym_electric_motor.visualization.motor_dashboard_plots import MeanEpisodeRewardPlot

    dashboard = MotorDashboard(
        state_plots = ['omega', 'i'],
        reward_plot = True,
        additional_plots=MeanEpisodeRewardPlot()
    )
    env = gem.make('my-env-id-v0', visualization=dashboard)

How To: Create your own plots
____________________________

