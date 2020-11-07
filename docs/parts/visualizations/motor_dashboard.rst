Motor Dashboard
###############################

..  contents::


..  toctree::
    :maxdepth: 1
    :caption: Available Plots:
    :glob:

    motor_dashboard_plots/*

Usage Guide
__________________
To use the dashboard, you have to import the class, define the plots,
instantiate the dashboard and pass it to the environment.

The most common plots can be quickly selected directly in the constructor of the dashboard.
Further plots like a *MeanEpisodeRewardPlot* or self-defined ones have to be instantiated and passed to the dashboard.

.. code-block:: python

    import gym_electric_motor as gem
    from gym_electric_motor.visualization import MotorDashboard
    from gym_electric_motor.visualization.motor_dashboard_plots import MeanEpisodeRewardPlot

    # create the dashboard and define the plots
    dashboard = MotorDashboard(
        state_plots = ['omega', 'i'], # Pass a list of state names or 'all' for all states to plot
        reward_plot = True, # True / False (False default)
        action_plots = [0] # 'all' plots all actions (if multiple are applied)
        additional_plots=[MeanEpisodeRewardPlot()] # Add all further plots here
    )

    # pass it to the environment
    env = gem.make('my-env-id-v0', visualization=dashboard)




Motor Dashboard API
__________________________

.. autoclass:: gym_electric_motor.visualization.motor_dashboard.MotorDashboard
   :members:
   :inherited-members:

..
    The following section is commented out. It may serve as a little introduction on how to define
    your own custom plots in the future.

    Create your own plots
    _____________________________
     In the following, there is
    .. code-block:: python

        import gym_electric_motor as gem
        from gym_electric_motor.visualization import MotorDashboard
        from gym_electric_motor.visualization.motor_dashboard_plots import TimePlot

        # the class may also derive from EpisodePlot or StepPlot, depending on the x-axis
        class MyPlot(TimePlot):
        """This plot will show the current step of the episode *k* on the y-axis.

         As it derives from TimePlot the x-Axis is the cumulative simulated time over all episodes.
         """

            def __init__(self):
                super().__init__()
                # Set the y-axis label
                self._label = 'k'

            def initialize(self, axis):
                super().initialize(axis)
                self._k_line, =  self._axis.plot([],[])
                self._lines.append(self._k_line)

            def set_env(self, env):
                super().set_env(env)
                self._k_data = np.ones_like(self._x_data) * np.nan
                self._y_data.append(self._k_data)

            def on_step_begin(self, k, action):
                super().on_step_begin(k, action)
                idx = self.data_idx
                self._k_data[idx] = k

            def _scale_y_axis(self):
                """This function can be defined to automatically scale the plots limits.
                Here, the y limits are set such that the data fits perfectly.
                """
                self._axis.set_ylim(0,max(self._k_data))

        dashboard = MotorDashboard(
            state_plots='all',
            additional_plots=[MyPlot()]
        )
        env = gem.make('my-env-id-v0', visualization=dashboard)
