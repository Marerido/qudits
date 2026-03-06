from __future__ import annotations

from typing import TYPE_CHECKING, cast
import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from ...compiler.compilation_minitools.local_compilation_minitools import regulate_theta

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class Rzz(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        parameters: list[int | float],
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ) -> None:

        # 🔹 Falls nur eine Dimension übergeben wird → automatisch [d, d]
        if isinstance(dimensions, int):
            dimensions = [dimensions, dimensions]

        if not isinstance(dimensions, list) or len(dimensions) != 2:
            raise TypeError("dimensions must be int or list[int, int]")

        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="rzz",
        )

        if self.validate_parameter(parameters):
            self.lev_a = cast(int, parameters[0])
            self.lev_b = cast(int, parameters[1])
            self.phi = regulate_theta(cast(float, parameters[2]))
            self._params = parameters

    def __array__(self) -> NDArray[np.complex128]:

        dim_ctrl, dim_target = self.dimensions
        dim_total = dim_ctrl * dim_target
        phi = self.phi

        result = np.zeros((dim_total, dim_total), dtype=np.complex128)

        # Für Qudits: Die Z-Matrix ist diagonal mit Werten (-1)^i für i=0,1,...,d-1
        # Aber in deinem Fall: Du willst zwischen zwei bestimmten Levels rotieren
        # Also: Z = diag(1, 1, ..., -1, ..., -1) mit lev_a Einträgen = 1 und restlichen = -1
        
        # Korrekte Implementierung:
        # Für ein Qudit der Dimension d: Z-Matrix ist diagonal mit Werten:
        # 1 für Zustände 0,1,...,lev_a-1
        # -1 für Zustände lev_a, lev_a+1, ..., d-1
        
        # Für das erste Qudit (Kontrolle)
        z_vals_ctrl = np.ones(dim_ctrl)
        for i in range(dim_ctrl):
            if i >= self.lev_a and i <= self.lev_b:
                z_vals_ctrl[i] = 1
            else:
                z_vals_ctrl[i] = -1
                
        # Für das zweite Qudit (Ziel)
        z_vals_target = np.ones(dim_target)
        for j in range(dim_target):
            if j >= self.lev_a and j <= self.lev_b:
                z_vals_target[j] = 1
            else:
                z_vals_target[j] = -1

        # Berechne die Matrix für Z ⊗ Z
        for i in range(dim_ctrl):
            for j in range(dim_target):
                # Z ⊗ Z Matrixelement: (Z ⊗ Z)|i⟩ ⊗ |j⟩ = z_vals_ctrl[i] * z_vals_target[j] * |i⟩ ⊗ |j⟩
                z_product = z_vals_ctrl[i] * z_vals_target[j]
                phase = np.exp(-1j * phi * z_product / 2)
                index = i * dim_target + j
                result[index, index] = phase

        return result

    def validate_parameter(self, parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            if len(parameter) != 3:
                return False

            assert isinstance(parameter[0], int)
            assert isinstance(parameter[1], int)
            assert isinstance(parameter[2], float)

            assert parameter[0] >= 0
            assert parameter[1] >= 0
            assert parameter[0] != parameter[1]

            return True

        return False

    @property
    def dimensions(self) -> list[int]:
        return cast(list[int], self._dimensions)