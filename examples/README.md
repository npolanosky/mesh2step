# Examples

Drop sample STL files here and convert them:

```bash
# Reconstructed (default): coplanar facets merged into single faces
python -m mesh2step bracket.stl -o bracket.step

# Compare against the classic faceted output
python -m mesh2step bracket.stl -o bracket_faceted.step --faceted
```

Inspect the result in FreeCAD or any STEP viewer and compare the face count:
a reconstructed prismatic part should have an order of magnitude fewer faces
than the faceted version.

To run under FreeCAD's own interpreter instead of your venv:

```bash
freecadcmd -c "import mesh2step.cli as c; raise SystemExit(c.main())" -- bracket.stl
```
