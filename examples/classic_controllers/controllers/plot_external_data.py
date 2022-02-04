def plot(external_reference_plots=(), state_names=(), external_plot=(), visualization=True, external_data=()):
    """
    This method passes the latest internally generated references of the controller the ExternalReferencePlots. The
    GEM-Environment uses this data to plot these references with the according states within its MotorDashboard.

    Args:
        external_reference_plots(Iterable[ExternalReferencedPlot]):
            The External Reference Plots that the internal reference data shall be passed to.
        state_names:
            The list of all environment state names.
        external_plot(Iterable[ExternalPlot])
        visualization:
            Boolean if there is a visualization.
        external_data:
            List of data to be plotted
    """

    if visualization:
        external_ref_plots = list(external_reference_plots)
        external_plots = list(external_plot)

        # Check, if the are external ref plots
        if len(external_ref_plots) != 0:
            # Read in the indices of the states and refrences
            ref_state_idxs = external_data['ref_state']

            plot_state_idxs = [
                list(state_names).index(external_ref_plot.state_plot) for external_ref_plot in external_reference_plots
            ]

            # Read in the data of the reference
            ref_values = external_data['ref_value']

            # Pass the values to the ExternallyReferencedStatePlot object
            for ref_state_idx, ref_value in zip(ref_state_idxs, ref_values):
                try:
                    plot_idx = plot_state_idxs.index(ref_state_idx)
                except ValueError:
                    pass    # Ignore reference input, if there is no reference in this plot
                else:
                    external_ref_plots[plot_idx].external_reference(ref_value)

        # Check if the are external plots and pass the data to the ExternalPlot object
        if len(external_plots) != 0:
            ext_state = external_data['external']
            for ext_plot, ext_data in zip(external_plots, ext_state):
                ext_plot.add_data(ext_data)
