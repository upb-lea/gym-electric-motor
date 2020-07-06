from .core import Callback
from gym_electric_motor.reference_generators import SubepisodedReferenceGenerator, SwitchedReferenceGenerator

class DummyCallback(Callback):
    def on_reset_begin(self):
        print('on_reset_begin')
    def on_reset_end(self):
        print('on_reset_end')
    def on_step_begin(self):
        print('on_step_begin')
    def on_step_end(self):
        print('on_step_end')
    def on_close(self):
        print('on_close')
        
class AdaptiveLimitMargin(Callback):
    def __init__(self, initial_limit_margin=(-0.1,0.1), maximum_limit_margin=(-1,1), step_size=0.1, update_time='episode', update_freq=10):
        assert update_time in ['step', 'episode'], "Chose an option of either 'step' or 'episode' for updating on cumultative steps or episodes"
        assert initial_limit_margin[1] > initial_limit_margin[0], "First element of limit margin has to be smaller than second"
        assert maximum_limit_margin[1] > maximum_limit_margin[0], "First element of limit margin has to be smaller than second"
        assert initial_limit_margin[0] >= -1, "Lower limit margin has to be bigger than or equal to -1"
        assert maximum_limit_margin[0] >= -1, "Lower limit margin has to be bigger than or equal to -1"
        assert initial_limit_margin[1] <= 1, "Upper limit margin has to be smaller than or equal to 1"
        assert maximum_limit_margin[1] <= 1, "Upper limit margin has to be smaller than or equal to 1"
        
        self._limit_margin = initial_limit_margin
        self._maximum_limit_margin = maximum_limit_margin
        self._step_size = step_size
        self._update_time = 0 if update_time == 'step' else 1
        if self._update_time == 0:
            self._step = 0
        else:
            self._episode = 0
        self._update_freq = update_freq
        
    def set_env(self, env):
        if env.reference_generator.__class__ == SwitchedReferenceGenerator:
            for sub_generator in env.reference_generator._sub_generators:
                assert issubclass(sub_generator.__class__, SubepisodedReferenceGenerator), ("The AdaptiveLimitMargin does only support the SubepisodedReferenceGenerator as reference generator or "
                                                                                            "SwitchedReferenceGenerator with SubepisodedReferenceGenerator as all sub reference generators")
        else:
            assert issubclass(env.reference_generator.__class__, SubepisodedReferenceGenerator), ("The AdaptiveLimitMargin does only support the SubepisodedReferenceGenerator as reference generator or" 
                                                                                                 "SwitchedReferenceGenerator with SubepisodedReferenceGenerator as all sub reference generators")
        self._env = env
        if env.reference_generator.__class__ == SwitchedReferenceGenerator:
            for sub_generator in self._env.reference_generator._sub_generators:
                sub_generator._limit_margin = self._limit_margin
        else:
            self._env.reference_generator._limit_margin = self._limit_margin
        
    def on_step_end(self):
        if self._update_time == 0:
            self._step += 1
            if self._step % self._update_freq == 0:
                self._step = 0
                self._update_limit_margin()     
                
    def on_reset_end(self):
        if self._update_time == 1:
            self._episode += 1
            if self._episode % self._update_freq == 0:
                self._episode = 0
                self._update_limit_margin()
                
    def _update_limit_margin(self):
        if self._limit_margin != self._maximum_limit_margin:
            new_lower_limit = self._limit_margin[0] - self._step_size
            new_lower_limit = new_lower_limit if new_lower_limit > self._maximum_limit_margin[0] else self._maximum_limit_margin[0]
            new_upper_limit = self._limit_margin[1] + self._step_size
            new_upper_limit = new_upper_limit if new_upper_limit < self._maximum_limit_margin[1] else self._maximum_limit_margin[1]
            self._limit_margin = (new_lower_limit, new_upper_limit)
            if self._env.reference_generator.__class__ == SwitchedReferenceGenerator:
                for sub_generator in self._env.reference_generator._sub_generators:
                    sub_generator._limit_margin = self._limit_margin
            else:
                self._env.reference_generator._limit_margin = self._limit_margin
                
        

    
    
    