from dataclasses import dataclass, fields

import torch


@dataclass
class HiddenToTransientParams:
    scan_coords: torch.Tensor
    scan_indices: torch.Tensor
    output_shape: tuple
    bin_length: float
    T_start: int
    T_end: int
    is_confocal: bool
    light_coord: torch.Tensor
    is_retroreflective: bool
    light_scan_is_reversed: bool
    falloff_scale: float = 1.0

    @classmethod
    def attributes(cls):
        return [x.name for x in fields(cls)]
