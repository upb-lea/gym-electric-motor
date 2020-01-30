from .wiener_process_reference_generator import WienerProcessReferenceGenerator
from .multi_reference_generator import MultiReferenceGenerator
from .zero_reference_generator import ZeroReferenceGenerator
from .sinusoidal_reference_generator import SinusoidalReferenceGenerator
from .step_reference_generator import StepReferenceGenerator
from .triangle_reference_generator import TriangularReferenceGenerator
from .sawtooth_reference_generator import SawtoothReferenceGenerator
from ..utils import register_class
from ..core import ReferenceGenerator

register_class(WienerProcessReferenceGenerator, ReferenceGenerator, 'WienerProcessReference')
register_class(MultiReferenceGenerator, ReferenceGenerator, 'MultiReference')
register_class(StepReferenceGenerator, ReferenceGenerator, 'StepReference')
register_class(SinusoidalReferenceGenerator, ReferenceGenerator, 'SinusReference')
register_class(TriangularReferenceGenerator, ReferenceGenerator, 'TriangleReference')
register_class(SawtoothReferenceGenerator, ReferenceGenerator, 'SawtoothReference')
