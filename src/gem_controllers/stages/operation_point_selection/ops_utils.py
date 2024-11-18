from .eesm_ops import EESMOperationPointSelection
from .extex_dc_ttc import ExtExDcOperationPointSelection
from .permex_dc_ops import PermExDcOperationPointSelection
from .pmsm_ops import PMSMOperationPointSelection
from .scim_ops import SCIMOperationPointSelection
from .series_dc_ops import SeriesDcOperationPointSelection
from .shunt_dc_ops import ShuntDcOperationPointSelection

torque_to_current_function = {
    "PermExDc": PermExDcOperationPointSelection,
    "ExtExDc": ExtExDcOperationPointSelection,
    "SeriesDc": SeriesDcOperationPointSelection,
    "ShuntDc": ShuntDcOperationPointSelection,
    "PMSM": PMSMOperationPointSelection,
    "SynRM": PMSMOperationPointSelection,
    "SCIM": SCIMOperationPointSelection,
    "EESM": EESMOperationPointSelection,
}
