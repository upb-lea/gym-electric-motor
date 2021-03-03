from gym_electric_motor.visualization.motor_dashboard_plots import StatePlot


class ExternallyReferencedStatePlot(StatePlot):

    def __init__(self, state):
        super().__init__(state)
        self.state_plot = state

    def set_reference(self, ref_states):
        self._referenced = self.state_plot in ref_states

    def set_env(self, env):
        super().set_env(env)

    def external_reference(self, value):
        self._ref_data[self.data_idx] = value * self._limits
