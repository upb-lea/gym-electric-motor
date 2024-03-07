from gym_electric_motor.visualization.motor_dashboard import StatePlot


class ReferencePlotter:
    """This class adds the reference values of the subordinate stages to the stage plots of the GEM environment."""

    def __init__(self):
        self._referenced_plots = []
        self._referenced_states = []
        self._maximum_reference = None
        self._plot_references = None
        self._maximum_reference_set = False

    def tune(self, env, referenced_states, plot_references, maximum_reference, **_):
        """
        Tune the reference plotter.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            referenced_states(np.ndarray): Array of all referenced states.
            plot_references(bool): Flag, if the references of the subordinate stages should be plotted.
            maximum_reference(dict): Dict containing all limited reference currents.

        """
        if plot_references:
            for visualization in env.visualizations:
                for time_plot in visualization._time_plots:
                    if isinstance(time_plot, StatePlot):
                        if time_plot.state in referenced_states:
                            self._referenced_plots.append(time_plot)
                            self._referenced_states.append(time_plot.state)

        for plot in self._referenced_plots:
            plot._referenced = True

        self._maximum_reference = maximum_reference

    def add_maximum_reference(self, state, value):
        self._maximum_reference[state] = value

    def update_plots(self, references):
        """
        Update the state plots of the GEM environment.

        Args:
            references(np.ndarray): Array of all reference values of the subordinate stages.
        """

        if not self._maximum_reference_set:
            for plot in self._referenced_plots:
                if plot.state in ["i_e", "i_a", "i"] and plot.state in self._maximum_reference.keys():
                    label = dict(
                        i="$i^*$$_{\mathrm{ max}}$", i_a="$i^*_{a}$$_\mathrm{ max}}$", i_e="$i^*_{e}$$_\mathrm{ max}}$"
                    )
                    plot._axis.axhline(self._maximum_reference[plot.state][0], c="g", linewidth=0.75, linestyle="--")
                    plot._axis.axhline(self._maximum_reference[plot.state][1], c="g", linewidth=0.75, linestyle="--")
                    labels = [legend._text for legend in plot._axis.get_legend().texts] + [label[plot.state]]
                    lines = plot._axis.lines[0 : len(labels) - 1] + plot._axis.lines[-1:]
                    plot._axis.legend(lines, labels, loc="upper left", numpoints=20)

            self._maximum_reference_set = True

        for plot, state in zip(self._referenced_plots, self._referenced_states):
            plot._ref_data[plot.data_idx] = references[state]
