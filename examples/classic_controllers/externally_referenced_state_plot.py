from gym_electric_motor.visualization.motor_dashboard_plots import StatePlot


class ExternallyReferencedStatePlot(StatePlot):
    """Plot that displays environments states together with externally generated references.

    These could be for example references that are generated intermediately within a cascaded controller.

    Usage Example
    -------------

    .. code-block:: python
        :emphasize-lines: 1,12

        my_externally_referenced_plot = ExternallyReferencedStatePlot(state='i_sd')

        env = gem.make(
            'DqCont-SC-PMSM-v0',
            visualization=dict(additional_plots=(my_externally_referenced_plot,),
        )
        done = True
        for _ in range(10000):
            if done:
                state, reference = env.reset()
            external_reference_value = my_external_isd_reference_generator.get_reference()
            my_externally_referenced_plot.external_reference(external_reference_value)
            env.render()
            action = env.action_space.sample()
            (state, reference), reward, done, _ = env.step(action)

    """

    def set_env(self, env):
        # Docstring from superclass
        super().set_env(env)

        # This plot is per definition always referenced
        self._referenced = True

    def external_reference(self, value):
        """Method to pass the externally generated reference.

        Arguments:
        ----------
        value: float
            The external reference value at the current time step.
        """
        self._ref_data[self.data_idx] = value * self._limits
