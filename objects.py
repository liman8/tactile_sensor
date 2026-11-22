"""
Object registry: objects.json, one entry per object of the tactile pipeline (multi-object run, results_multi/).

    import objects
    objects.names()                       # ["fork_stl", "fork2_stl", "knife_stl", "spoon_stl"] (registry order = id)
    objects.get("knife_stl")              # the entry (dict)
    objects.part_names("spoon_stl")       # ["bowl", "neck", "handle"]
    objects.part_of("fork_stl", x_mm, nz) # part index per point (len(names) = "edge" when |nz| < edge_nz)

Every entry: the STL file and the units it was read in (and why), the scene XML that contains the object, the mesh
folder made by stl_to_mjcf.py, material / density and the mass and centre of mass computed from the mesh, the CoACD
threshold, the mirror symmetry about the long axis (STL x) with its Hausdorff distance, whether "face up / face down"
is a meaningful label, the parts along the long axis, and where the stable resting poses are.

Parts: names[k] covers x >= x_starts_mm[k] (STL / body frame x, mm; starts in decreasing order) and below the previous
start; x below the last start is the last name. A surface point whose |normal . z| < edge_nz is "edge" (index
len(names)), as collect_dataset.py has always done for fork_stl. fork_stl's values (7.0, 1.0, -15.0, 0.5) are the
constants collect_dataset.py and train_centre.py used before the registry existed, so their outputs are unchanged.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "objects.json")
_REG = None


def load() -> dict:
    global _REG
    if _REG is None:
        with open(REGISTRY) as f:
            _REG = json.load(f)
    return _REG


def names() -> list:
    return list(load()["objects"])


def get(name: str) -> dict:
    return load()["objects"][name]


def object_id(name: str) -> int:
    return names().index(name)


def part_names(name: str) -> list:
    return list(get(name)["parts"]["names"])


def part_starts_mm(name: str) -> list:
    return [float(x) for x in get(name)["parts"]["x_starts_mm"]]


def edge_nz(name: str) -> float:
    return float(get(name)["parts"]["edge_nz"])


def part_from_x(starts, x_mm):
    """Part index along the long axis for the given starts (no edge rule); int64, like the old nested np.where."""
    x = np.asarray(x_mm)
    p = np.full(np.shape(x), len(starts))
    for k in reversed(range(len(starts))):
        p = np.where(x >= starts[k], k, p)
    return p


def part_of(name: str, x_mm, nz=None):
    """Part index per point (int8): along the long axis, or "edge" (len(names)) where |nz| < edge_nz."""
    p = part_from_x(part_starts_mm(name), np.asarray(x_mm, float))
    if nz is not None:
        p = np.where(np.abs(np.asarray(nz, float)) < edge_nz(name), len(part_names(name)), p)
    return p.astype(np.int8)


def scene_xml(name: str) -> str:
    return os.path.join(HERE, get(name)["scene"])


def mesh_dir(name: str) -> str:
    return os.path.join(HERE, get(name)["mesh_dir"])
