from ...core import ElectricMotorEnvironment
from gym_electric_motor.physical_systems.physical_systems import SwitchedReluctanceMotorSystem
from ...reference_generators import WienerProcessReferenceGenerator
from ...reward_functions import WeightedSumOfErrors
from ...constraints import SquaredConstraint


class SwitchedReluctanceMotorEnvironment(ElectricMotorEnvironment):

    def __init__(self, motor='SRM', reward_function=None, reference_generator=None, constraints=None, **kwargs):
        """
        Args:
            motor(ElectricMotor): Electric Motor used in the PhysicalSystem
            reward_function(RewardFunction): Reward Function for the environment
            reference_generator(ReferenceGenerator): Reference Generator for the environment
            kwargs(dict): Further kwargs tot pass to the superclass and the submodules
        """
        physical_system = SwitchedReluctanceMotorSystem(motor=motor, **kwargs)
        reference_generator = reference_generator or WienerProcessReferenceGenerator(**kwargs)
        reward_function = reward_function or WeightedSumOfErrors(**kwargs)
        constraints_ = constraints if constraints is not None \
            else tuple(physical_system._electrical_motor.CURRENTS)
        super().__init__(
            physical_system, reference_generator=reference_generator, reward_function=reward_function,
            constraints=constraints_, **kwargs
        )


class DiscSwitchedReluctanceMotorEnvironment(SwitchedReluctanceMotorEnvironment):
    """
    write me
    """
    # subconverters not defined => will be dynamically set in physicalSystem according to stator pole pairs
    # if specified by the user, physicalSystem will check if valid
    # default is 4QC
    def __init__(self, tau=1e-5, converter='Disc-Multi', **kwargs):
        super().__init__(tau=tau, converter=converter, **kwargs)


class ContSwitchedReluctanceMotorEnvironment(SwitchedReluctanceMotorEnvironment):
    """
    write me
    """
    # subconverters not defined => will be dynamically set in physicalSystem according to stator pole pairs
    # if specified by the user, physicalSystem will check if valid
    # default is 4QC
    def __init__(self, tau=1e-5, converter='Cont-Multi', **kwargs):
        super().__init__(tau=tau, converter=converter, **kwargs)
