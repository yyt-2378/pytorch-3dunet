"""
Utility script to build a toy HDF5 volume for super-resolution + optional material conditioning.

The generated file contains:
- raw: low-resolution input volume shaped (C, Z, Y, X)
- label: target high-resolution volume with the same spatial dimensions
- condition: optional per-voxel condition channels (e.g., element masks) shaped (K, Z, Y, X)

This matches the expectations of pytorch-3dunet's HDF5Dataset, which requires that
condition maps share the same spatial shape as raw/label.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


ELEMENTS = ("Ga", "Sr", "O", "Vacancy")


def build_condition_map(volume_shape: tuple[int, int, int]) -> np.ndarray:
    """Create a simple per-voxel one-hot mask for each element channel."""
    z, y, x = np.indices(volume_shape)
    cond = np.zeros((len(ELEMENTS),) + volume_shape, dtype=np.float32)

    cond[0] = (z % 4 == 0).astype(np.float32)  # Ga stripes
    cond[1] = (y % 5 == 0).astype(np.float32)  # Sr stripes
    cond[2] = (x % 6 == 0).astype(np.float32)  # O stripes
    cond[3] = ((z + y + x) % 9 == 0).astype(np.float32)  # vacancy marker
    return cond


def build_pair(volume_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one low-res/high-res pair plus condition channels."""
    # high-res target: smooth spheres
    z, y, x = np.indices(volume_shape)
    center = np.array(volume_shape)[:, None, None, None] / 2
    radius = min(volume_shape) / 3
    dist = np.sqrt(((np.stack([z, y, x]) - center) ** 2).sum(0))
    label = np.exp(-(dist**2) / (2 * (radius / 2) ** 2)).astype(np.float32)[None]

    # low-res input: blurred + downscaled noise added back in
    raw = label + 0.1 * np.random.randn(*label.shape).astype(np.float32)
    raw = raw.astype(np.float32)

    condition = build_condition_map(volume_shape)
    return raw, label, condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("toy_superres_condition.h5"), help="HDF5 path to write")
    parser.add_argument("--depth", type=int, default=16, help="Z-depth of the volume")
    parser.add_argument("--height", type=int, default=48, help="Y dimension of the volume")
    parser.add_argument("--width", type=int, default=48, help="X dimension of the volume")
    args = parser.parse_args()

    volume_shape = (args.depth, args.height, args.width)
    raw, label, condition = build_pair(volume_shape)

    with h5py.File(args.output, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
        f.create_dataset("condition", data=condition, compression="gzip")

    print(f"Wrote sample HDF5 with raw/label/condition to {args.output}")
    print(f"raw shape: {raw.shape}, label shape: {label.shape}, condition shape: {condition.shape}")
    print(f"condition channels: {ELEMENTS}")


if __name__ == "__main__":
    main()
