from ..core import ReferenceGenerator
from ..utils import register_class
from .const_reference_generator import ConstReferenceGenerator
from .laplace_process_reference_generator import LaplaceProcessReferenceGenerator
from .multiple_reference_generator import MultipleReferenceGenerator
from .sawtooth_reference_generator import SawtoothReferenceGenerator
from .sinusoidal_reference_generator import SinusoidalReferenceGenerator
from .step_reference_generator import StepReferenceGenerator
from .subepisoded_reference_generator import SubepisodedReferenceGenerator
from .switched_reference_generator import SwitchedReferenceGenerator
from .triangle_reference_generator import TriangularReferenceGenerator
from .wiener_process_reference_generator import WienerProcessReferenceGenerator
from .zero_reference_generator import ZeroReferenceGenerator

register_class(WienerProcessReferenceGenerator, ReferenceGenerator, "WienerProcessReference")
register_class(SwitchedReferenceGenerator, ReferenceGenerator, "SwitchedReference")
register_class(StepReferenceGenerator, ReferenceGenerator, "StepReference")
register_class(SinusoidalReferenceGenerator, ReferenceGenerator, "SinusReference")
register_class(TriangularReferenceGenerator, ReferenceGenerator, "TriangleReference")
register_class(SawtoothReferenceGenerator, ReferenceGenerator, "SawtoothReference")
register_class(ConstReferenceGenerator, ReferenceGenerator, "ConstReference")
register_class(MultipleReferenceGenerator, ReferenceGenerator, "MultipleReference")
register_class(SubepisodedReferenceGenerator, ReferenceGenerator, "SubepisodedReference")
