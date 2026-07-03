"""Provision tests (no FreeCAD/network): the prep-dep install must run the
required (manifold3d) and optional (pymeshlab) package groups as SEPARATE pip
invocations, so a failing pymeshlab wheel can't drop manifold3d.
"""

from __future__ import annotations

from mesh2step import provision


def test_prep_deps_install_required_and_optional_separately(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_install(freecad_python, target, packages, log=None):
        calls.append(list(packages))
        # Simulate the optional (pymeshlab) group FAILING while the required
        # (manifold3d) group succeeds.
        return "manifold3d" in " ".join(packages)

    monkeypatch.setattr(provision, "_pip_install_into", fake_install)
    monkeypatch.setattr(provision, "pydeps_dir", lambda fc: tmp_path)
    monkeypatch.setattr(provision, "prep_deps_present",
                        lambda fc: True)  # manifold3d imports after install
    monkeypatch.setattr(provision, "pymeshlab_usable", lambda fc: False)
    monkeypatch.setattr(provision, "_purge_shadowing_numpy", lambda t, log=None: False)

    result = provision.ensure_prep_deps("/fake/python", force=True)

    # Two independent pip invocations, one per group.
    assert len(calls) == 2
    assert any("manifold3d" in " ".join(pkgs) for pkgs in calls)
    assert any("pymeshlab" in " ".join(pkgs) for pkgs in calls)
    # No single call mixes the required and optional packages (would fail atomically).
    for pkgs in calls:
        joined = " ".join(pkgs)
        assert not ("manifold3d" in joined and "pymeshlab" in joined)
    # manifold3d succeeded, so the deps dir is still returned despite pymeshlab failing.
    assert result == tmp_path


def test_prep_deps_required_group_is_manifold3d_only():
    assert provision.REQUIRED_PACKAGES == ["manifold3d>=3.0"]
    assert all("pymeshlab" in p for p in provision.OPTIONAL_PACKAGES)
    assert provision.REQUIRED_MODULE == "manifold3d"
