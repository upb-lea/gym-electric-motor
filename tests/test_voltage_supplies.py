import gym_electric_motor.envs
from gym_electric_motor.physical_systems.voltage_supplies import *
import pytest


def test_voltage_supply_default():
    u_default = 600
    # default initialization
    voltage_supply = IdealVoltageSupply()
    assert voltage_supply.u_nominal == u_default
    assert voltage_supply.get_voltage() == u_default
    assert voltage_supply.reset() == u_default


def test_voltage_supply_with_parameter():
    u_n = 450
    # initialization with parameter
    voltage_supply = IdealVoltageSupply(u_n)
    assert voltage_supply.u_nominal == u_n
    assert voltage_supply.get_voltage() == u_n
    assert voltage_supply.reset() == u_n
