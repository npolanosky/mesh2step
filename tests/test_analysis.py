"""Bounding-box and unit-scaling tests (numpy only, no FreeCAD)."""

from __future__ import annotations

import numpy as np
import pytest

from mesh2step.analysis import axis_aligned_bbox, measure, oriented_bbox
from mesh2step.config import ConversionConfig


def test_aabb_dimensions(cube_triangles):
    verts = cube_triangles.reshape(-1, 3)
    box = axis_aligned_bbox(verts)
    assert np.allclose(box.dimensions, [10, 10, 10])
    assert box.volume == pytest.approx(1000.0)


def test_obb_matches_aabb_for_axis_aligned_cube(cube_triangles):
    verts = cube_triangles.reshape(-1, 3)
    obb = oriented_bbox(verts)
    # A cube's oriented box is the same size as its axis-aligned box.
    assert np.allclose(np.sort(obb.dimensions), [10, 10, 10])


def test_obb_recovers_rotated_box_size():
    # A 30x10x5 box rotated 30 deg about Z: the OBB should recover 30,10,5.
    pts = np.array(np.meshgrid([0, 30], [0, 10], [0, 5])).T.reshape(-1, 3).astype(float)
    a = np.radians(30)
    rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rotated = pts @ rot.T
    obb = oriented_bbox(rotated)
    assert np.allclose(np.sort(obb.dimensions)[::-1], [30, 10, 5], atol=1e-6)


def test_measure_reports_both_boxes(cube_triangles):
    m = measure(cube_triangles.reshape(-1, 3))
    assert "aabb" in m and "obb" in m
    assert m["aabb"]["dimensions"] == pytest.approx([10, 10, 10])


@pytest.mark.parametrize(
    "units,factor", [("mm", 1.0), ("cm", 10.0), ("m", 1000.0), ("in", 25.4)]
)
def test_unit_scale_presets(units, factor):
    assert ConversionConfig(source_units=units).scale_to_mm == factor


def test_scale_override_wins():
    cfg = ConversionConfig(source_units="m", scale_override=2.0)
    assert cfg.scale_to_mm == 2.0


def test_unknown_units_raise():
    with pytest.raises(ValueError):
        ConversionConfig(source_units="furlong").scale_to_mm
