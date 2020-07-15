from gym_electric_motor.reference_generators import SubepisodedReferenceGenerator, SwitchedReferenceGenerator
from gym_electric_motor.callbacks import RampingLimitMargin
from tests.testing_utils import DummyElectricMotorEnvironment, DummyReferenceGenerator
import pytest

class TestRampingLimitMargin:
    test_class = RampingLimitMargin
    key = ''

    def test_update(self):
        #step updates
        callback = self.test_class(initial_limit_margin=(-0.1,0.1), maximum_limit_margin=(-1,1), step_size=0.1, update_time='step', update_freq=100)
        callbacks = [callback]
        env = DummyElectricMotorEnvironment(reference_generator=SubepisodedReferenceGenerator(), callbacks=callbacks)
        #initial limit margin set
        assert env._reference_generator._limit_margin == (-0.1,0.1)
        #no updates after episodes
        for i in range(1000):
            env.reset()
        assert env._reference_generator._limit_margin == (-0.1,0.1)
        #updates after steps
        for i in range(99):
            env.step()
        assert env._reference_generator._limit_margin == (-0.1,0.1)
        env.step()
        assert env._reference_generator._limit_margin == (-0.2,0.2)
        #limit margin does not exceed maximum
        for i in range(10000):
            env.step()
        assert env._reference_generator._limit_margin == (-1,1)

        #episode updates
        callback = self.test_class(initial_limit_margin=(0.0,0.3), maximum_limit_margin=(0,1), step_size=0.2, update_time='episode', update_freq=20)
        callbacks = [callback]
        env = DummyElectricMotorEnvironment(reference_generator=SubepisodedReferenceGenerator(), callbacks=callbacks)
        #initial limit margin set
        assert env._reference_generator._limit_margin == (0.0,0.3)
        #no updates after steps
        for i in range(1000):
            env.step()
        assert env._reference_generator._limit_margin == (0.0,0.3)
        #updates after episodes
        for i in range(19):
            env.reset()
        assert env._reference_generator._limit_margin == (0.0,0.3)
        env.reset()
        assert env._reference_generator._limit_margin == (0.0,0.5)
        #limit margin does not exceed maximum
        for i in range(1000):
            env.reset()
        assert env._reference_generator._limit_margin == (0,1)
    
    def test_initial_values(self):
        callback = self.test_class()
        assert callback._limit_margin == (-0.1,0.1)
        assert callback._maximum_limit_margin == (-1,1)
        assert callback._step_size == 0.1
        assert callback._update_time == 'episode'
        assert callback._update_freq == 10

    def test_update_switched(self):
        #since general update behavior is tested before now we just want to take a short look at switched update behavior
        callback = self.test_class()
        callbacks = [callback]
        sub_generators = [SubepisodedReferenceGenerator(),SubepisodedReferenceGenerator(),SubepisodedReferenceGenerator()]
        switched = SwitchedReferenceGenerator(sub_generators)
        env = DummyElectricMotorEnvironment(reference_generator=switched, callbacks=callbacks)
        # all sub generators get initial limit margin
        for sub_generator in sub_generators:
            assert sub_generator._limit_margin == (-0.1,0.1)
        #all sub generators get updated
        for i in range(10):
            env.reset()
        for sub_generator in sub_generators:
            assert sub_generator._limit_margin == (-0.2,0.2)
            
    def test_right_reference(self):
        callback = self.test_class()
        callbacks = [callback]
        #reference generator has to be a subclass of SubepisodedReferenceGenerator
        with pytest.raises(AssertionError) as excinfo:
            env = DummyElectricMotorEnvironment(reference_generator=DummyReferenceGenerator(), callbacks=callbacks)
        assert "The RampingLimitMargin does only support" in str(excinfo.value)
        
        # all sub generators have to subclasses of SubepisodedReferenceGenerator
        sub_generators = [SubepisodedReferenceGenerator(),SubepisodedReferenceGenerator(),DummyReferenceGenerator()]
        switched = SwitchedReferenceGenerator(sub_generators)
        with pytest.raises(AssertionError) as excinfo:
            env = DummyElectricMotorEnvironment(reference_generator=switched, callbacks=callbacks)
        assert "The RampingLimitMargin does only support" in str(excinfo.value)

        




            
        
        