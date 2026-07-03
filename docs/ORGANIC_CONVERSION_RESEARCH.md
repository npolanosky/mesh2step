# Organic / Freeform Mesh → Smooth B-rep: Research Survey & Architecture Recommendation

Status: research complete (2026-07-03), pre-design. Scope: how professional
tools convert organic triangle meshes (curved shells, sculpted surfaces,
ergonomic shapes) into smooth B-rep solids, and which approach mesh2step can
concretely implement with Python + FreeCAD/OpenCASCADE + numpy (pymeshlab and
manifold3d in subprocess; new pip deps acceptable if wheels exist for
macOS arm64 + Windows; project license MIT).

Verification discipline: every load-bearing claim carries a source URL.
Claims that could not be verified against a primary source are marked
**[UNVERIFIED]** or **[snippet-only]** (search-engine snippet of a page that
blocked full fetch).

---

## Part I — How professional tools do it

### 1. Autodesk Fusion 360 "Convert Mesh" (Organic / T-Splines)

**Pipeline.** Fusion's Convert Mesh has three methods
([Autodesk Help](https://help.autodesk.com/view/fusion360/ENU/?guid=MESH-CONVERT-TO-SOLID)):

- **Faceted** — one B-rep face per triangle (our current organic fallback,
  exactly).
- **Prismatic** — merges facet groups into analytic faces (our current
  analytic tier, exactly).
- **Organic** — requires the Product Design Extension; produces a **T-spline
  body** which becomes a NURBS B-rep on Finish Form. Resolution is
  controlled "By Accuracy" (Low→Precise) or "By Face Number" (target quad
  count; tutorials default to ~100 faces —
  [Product Design Online](https://productdesignonline.com/how-to-convert-a-3d-scan-to-t-splines-in-fusion-360-kevin-kennedy/)).

The internal chain is **triangle mesh → quad remesh → T-spline control cage →
(on finish) trimmed-NURBS B-rep**. Autodesk never states this verbatim in one
document; it is inferred from separately documented behaviors (remesh on
Face Count [snippet-only, [Fusion blog](https://www.autodesk.com/products/fusion-360/blog/mesh-into-solid-body-conversion/)];
T-Spline Form in timeline; NURBS on Finish Form). **[Pipeline shape verified
piecewise, not as a single primary statement.]**

**T-splines background.** Autodesk acquired T-Splines Inc. (founded by
Thomas Sederberg, BYU) in Dec 2011
([Autodesk press release](https://investors.autodesk.com/news-releases/news-release-details/autodesk-acquires-t-splines-modeling-technology-assets)).
A T-spline generalizes NURBS by letting control-point rows terminate
(T-junctions), enabled by per-control-point knot vectors → local refinement
without full isocurve propagation ([Wikipedia](https://en.wikipedia.org/wiki/T-spline)).
T-NURCCs (the SIGGRAPH 2003 paper's full construct —
[Sederberg et al., "T-splines and T-NURCCs", ACM TOG 22(3)](https://dl.acm.org/doi/abs/10.1145/1201775.882295))
are a superset of both NURBS and Catmull-Clark subdivision surfaces, adding
extraordinary (star) points for arbitrary topology. The core T-splines patent
US 7,274,364 **expired in 2024** ([Wikipedia](https://en.wikipedia.org/wiki/T-spline)).
T-spline → NURBS is exact away from star points (Bézier/knot-insertion
extraction) but **decomposes into many trimmed patches and degrades near
star points** ("small ripples near extraordinary points can turn into zebra
noise" — [Novedge history essay](https://novedge.com/blogs/design-news/design-software-history-continuity-in-freeform-cad-why-nurbs-patchcraft-gave-way-to-subdivision-and-t-splines)).

**Documented failure modes on real scans** (why users call it buggy):

- **Triangle/n-gon density**: refuses with "This body contains a high
  percentage of triangles or n-sided polygons"; Autodesk's own workaround is
  Faceted conversion ([cadforum.cz](https://www.cadforum.cz/en/conversion-of-mesh-to-t-spline-fails-in-fusion360-tip10537)).
  Soft ">10,000 triangles" slow-performance warning; community practice is
  to decimate to ≤~100k facets first. No hard published cap (a rumored ~50k
  limit was **not confirmed** anywhere).
- **Self-intersections**: "T-spline model failed to convert" / "T-spline
  surface self-intersects" blocks B-rep output — and self-intersections are
  frequently *introduced by the remesh step itself* on curved regions.
  Repair Body often only flags, not fixes, them
  ([Autodesk KB, snippet-only](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/T-Spline-model-failed-to-convert-in-Fusion-360.html);
  [Product Design Online repair guide](https://productdesignonline.com/tips-and-tricks/how-to-repair-self-intersecting-t-spline-errors-in-fusion-360/)).
- **Topology errors / star-point pileups**: quad meshes with many poles
  (extraordinary vertices) show as red error points and won't convert
  [snippet-only, Autodesk forums]. Non-manifold input, open boundaries,
  floating shells all break it [snippet-only].
- **Quality trade-off even on success**: "either lose a lot of the smaller
  edge details, or your face count is too high" — community consensus is
  it's not a high-quality path for scans.

**Takeaway for us:** Fusion's organic converter fails on exactly the inputs
mesh2step already sanitizes (manifold3d union for self-intersections,
pymeshlab decimation for density). Its two conversion methods, Prismatic and
Organic, are *separate modes the user must choose*; a hybrid
prismatic+organic segmentation is not offered. That is our differentiator.

### 2. Rhino: QuadRemesh → SubD → ToNURBS (the reference pipeline)

**QuadRemesh algorithm.** The widespread "Rhino QuadRemesh = Instant Meshes"
belief is a **myth for the production feature**. McNeel's Brian Gillespie:
"We licensed a solution for this that works well, but is completely opaque to
us" ([McNeel forum](https://discourse.mcneel.com/t/v7-quadremesh-algorithm/154314)).
The myth's origin: McNeel's earlier free **CreateQuadMesh** plug-in for
Rhino 6 really did wrap **Instant Meshes + QuadriFlow** (Dale Fugier:
"a simple, free plug-in… that uses these two utilities" —
[McNeel forum](https://discourse.mcneel.com/t/create-quad-meshes-plug-in-available-on-food4rhino/75982)).
The production QuadRemesh is credited to "@maxime1"
([McNeel forum](https://discourse.mcneel.com/t/rhino-7-feature-quadremesh/85601/33));
Maxime Rouca is the author of Exoside QuadRemesher and ZBrush's ZRemesher
([CG Channel](https://www.cgchannel.com/2019/10/check-out-neat-automatic-retopology-tool-quadremesher/)),
so the licensed algorithm is very likely the ZRemesher/Exoside lineage —
**[UNVERIFIED: no primary source states the identity outright]**. Either
way, the *algorithm class* is curvature/feature-guided field-aligned quad
retopology, and McNeel itself shipped the Instant Meshes/QuadriFlow variant
as good-enough — which is what's available to us permissively.

**SubD.** Rhino SubD is **Catmull-Clark**, stated plainly in the openNURBS
developer guide ("high-precision Catmull-Clark subdivision surfaces";
`ON_SubD`, limit surface via `ON_SubD::GetSurfaceBrep()` —
[developer.rhino3d.com](https://developer.rhino3d.com/guides/opennurbs/reading-subdivision-surfaces/)).
McNeel: "Rhino evaluation results comply 100% with public domain algorithms
widely described in published technical literature." I.e. **the math of the
best-in-class pipeline is public domain**.

**ToNURBS.** Two patch-mapping modes
([ToNURBS command doc](http://docs.mcneel.com/rhino/8/help/en-us/commands/tonurbs.htm)):

- **Unpacked** — one NURBS surface per quad SubD face (bicubic; "untrimmed
  per quad" is the evident behavior but the word "untrimmed" is not in the
  doc — **[UNVERIFIED verbatim]**).
- **Packed** — merges regular quad grids into larger single NURBS surfaces
  (fewer patches).

Continuity: **G2 in regular regions** (the Catmull-Clark limit surface is C²
away from extraordinary vertices). At **extraordinary vertices (valence ≠ 4)**
exact bicubic representation is impossible, so ToNURBS offers approximation
levels **G0 / G1 / G1x / G1xx / G2** — G2 "adds bulges or dents around
extraordinary vertices" — producing fanned multi-patch topology around each
star point. Documented practical limitations
([Novedge best-practices writeup](https://novedge.com/blogs/design-news/rhino-3d-tip-subd-to-nurbs-best-practices-for-cad-cam-bim-handoffs)):
patch explosion / heavy files, visible seams under zebra analysis at star
points, tiny gaps needing `ShowEdges` hunts, downstream import errors at
tight tolerances. Universal mitigation: **minimize extraordinary vertices**
(all-quad, valence-4-dominant remesh) — conversion quality is governed by
quad-mesh topology, not the converter.

"Best-in-class" status: widely repeated, **no independent benchmark found**
— treat as practitioner consensus, not a measured fact.

### 3. Geomagic Design X / Wrap, SpaceClaim, QuickSurface, ZBrush

**Geomagic Design X "Auto Surface" (exact surfacing)** is the other
professional architecture — *patch network + per-patch fit* instead of
subdivision. Four stages
([LaserDesign step-by-step](https://www.laserdesign.com/4-easy-steps-to-nurbs-surface-modeling-in-geomagic-design-x/);
official [Autosurface tutorial page](https://support.geomagic.com/s/article/Autosurface)
exists but its body failed to load — **[secondary sources only, mutually
consistent]**):

1. **Extract Contours** — auto-detect curvature/feature lines on the scan.
2. **Apply Patchwork** — construct a quad patch network over the whole mesh,
   patch edges following the contours.
3. **Repair Patches** — fix intersecting paths, poor patch angles,
   high-degree corners, high-deviation patches.
4. **Fit Surface Patches** — least-squares NURBS per patch, knit; watertight
   mesh → closed solid; export STEP/IGES/Parasolid.

Output is a **"dumb solid"** (no feature tree). Vendor guidance is explicitly
**hybrid**: auto-surface the organic regions, model mechanical features
parametrically, boolean them together
([Hawk Ridge hybrid-modeling guide](https://hawkridgesys.com/blog/hybrid-modeling-with-geomagic-design-x)) —
independent validation of mesh2step's hybrid-segmentation thesis.

**Ansys SpaceClaim Skin Surface** is the manual end of the same idea: user
sketches patch boundaries on the facet mesh, tool fits and stitches
([SpaceClaim help](http://help.spaceclaim.com/2016.1.0/en/Content/SkinSurface.htm)).
**QuickSurface** claims full-auto freeform STL→NURBS with manual
control-point-resolution override and live deviation maps
([quicksurface.com](https://www.quicksurface.com/step-by-step-cad-modeling-from-freeform-3d-scans/)).
**ZBrush ZRemesher** is a proprietary curvature/feature-guided quad
retopologizer in the same broad class as field-aligned methods; Maxon does
not disclose whether it computes an explicit cross-field
([Maxon docs](https://help.maxon.net/zbr/en-us/Content/html/user-guide/3d-modeling/topology/zremesher/zremesher.html);
field-alignment claim is practitioner inference — **[UNVERIFIED]**).

**The published algorithm behind "patch network + continuity-constrained
fitting":**

- **US Patent 10,296,664 (Autodesk)** — "Scalable and precise fitting of
  NURBS surfaces to large-size mesh representations": segment mesh into
  rectangular (u,v) patches → per-patch weighted least-squares NURBS fit →
  vertex/edge constraints so neighbors meet with continuity
  ([Google Patents](https://patents.google.com/patent/US10296664B2/en)).
  Granted 2019; with continuations 10,614,178 and 11,263,356. **These are
  live patents (in force into the 2030s)** — a freedom-to-operate concern
  for any implementation that closely follows the rectangular-grid
  segmentation + constrained-LSQ recipe. (Exact continuity order claimed,
  G1 vs C², not fully confirmed from claim text — **[VERIFY if pursued]**.)
- Academic antecedents: "Adaptive patch-based mesh fitting for reverse
  engineering" (CAD 2007,
  [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0010448507002205),
  **[abstract-only]**); "Fast and accurate NURBS fitting for reverse
  engineering" ([Springer 2011](https://link.springer.com/article/10.1007/s00170-010-2947-1)).
- **PolyWorks|Modeler** documents fitting a network of G1-continuous NURBS
  over a curve network with T-junction support
  ([polyworks.com](https://www.polyworks.com/en-us/products/polyworks-modeler)).

**Industry reality check:** auto-surfacing is reliable for a watertight
organic *wrap*, but engineering-grade output still expects
human-in-the-loop patch layout and mandatory deviation verification
([Formlabs RE guide](https://formlabs.com/blog/how-to-use-3d-scanning-and-3d-printing-for-reverse-engineering/);
[GoEngineer on Wrap](https://www.goengineer.com/reverse-engineering/geomagic-wrap)).
For mesh2step this calibrates expectations: an automatic organic tier should
target "smooth, watertight, dimensionally honest within a reported
tolerance," not "designer-quality patch layout."

---

## Part II — Building blocks (algorithms, licenses, wheels)

### 4. Quad remeshing

| Tool | Algorithm | License | Python / wheels (macOS arm64 + Windows) |
|---|---|---|---|
| **Instant Meshes** ([repo](https://github.com/wjakob/instant-meshes)) | Local orientation-field + position-field smoothing, quad-dominant extraction (Jakob et al., [SIGGRAPH Asia 2015](https://rgl.epfl.ch/publications/Jakob2015Instant)) | **BSD-3** ✅ ([LICENSE](https://raw.githubusercontent.com/wjakob/instant-meshes/master/LICENSE.txt)) | No official binding; CLI subprocess. **But see pynanoinstantmeshes below.** |
| **pynanoinstantmeshes** ([PyPI](https://pypi.org/project/pynanoinstantmeshes/), [repo](https://github.com/vork/PyNanoInstantMeshes)) | Binding of a nano re-implementation of Instant Meshes; numpy in/out | **BSD** ✅ | **Real binary wheels: macOS arm64 ✅, Windows x64 ✅, manylinux ✅** (py3.8–3.12). Verified on PyPI. Small project (maturity risk); output is quad-dominant, not guaranteed pure-quad. |
| **QuadriFlow** ([repo](https://github.com/hjwdzh/QuadriFlow), [paper](http://stanford.edu/~jingweih/papers/quadriflow/quadriflow.pdf)) | Instant-Meshes framework + **global min-cost-flow** to remove position-field singularities → clean **manifold pure-quad** output; SAT option guarantees watertight ([repo README](https://github.com/hjwdzh/QuadriFlow)) | **MIT** ✅ (audit vendored Eigen's optional LGPL parts) | No official binding. `pyQuadriFlow` ([PyPI](https://pypi.org/project/pyQuadriFlow/)) is a **`py3-none-any` wheel with `license: null`** — no real compiled binaries for our platforms and unlicensed: **avoid**. Correct path: build the MIT CLI ourselves (CMake, small) and shell out — same pattern Blender uses (`extern/quadriflow`). |
| **libigl** ([PyPI](https://pypi.org/project/libigl/)) | Cross-field building blocks; full MIQ quadrangulation is **C++-only** (and drags CoMISo, LGPL/GPL) — Python bindings expose `comb_cross_field`, `cross_field_mismatch`, `lscm`, `harmonic`, but **no `miq`/`nrosy`/quad extraction** (verified against [binding docs](https://libigl.github.io/libigl-python-bindings/)) | MPL2 ✅ | **Real wheels: macosx_11_0_arm64 ✅, win_amd64 ✅** (v2.6.x). Useful for parametrization (harmonic/LSCM), not for turnkey quad remeshing. |

Fewer extraordinary vertices ⇒ better NURBS output (Part I). QuadriFlow's
global singularity removal is exactly this knob; Instant Meshes is faster
but leaves more irregular vertices.

### 5. Catmull-Clark → NURBS extraction (the math)

- **Stam 1998**, "Exact Evaluation of Catmull-Clark Subdivision Surfaces at
  Arbitrary Parameter Values" ([PDF](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/sig98.pdf)):
  Catmull-Clark generalizes the uniform bicubic B-spline; **every interior
  quad face whose four vertices are all valence-4 corresponds exactly to one
  uniform bicubic B-spline patch defined by its 4×4 control-point
  neighborhood**. No approximation. Near an extraordinary vertex, the limit
  surface is evaluated exactly via eigendecomposition of the subdivision
  matrix — exact points/normals, but as an infinite nested patch sequence,
  not a finite NURBS.
- **Each subdivision step isolates extraordinary vertices** (new vertices
  are valence 4), so after 1–2 Catmull-Clark refinements almost all faces
  are regular ⇒ exact bicubic patches; the irregular region shrinks
  geometrically ("refine-and-cap").
- **Extraordinary-vertex (EV) patch options:**
  1. **Refine-and-cap**: subdivide until the EV neighborhood is below
     tolerance, cap the remaining n-sided hole with an approximating patch
     (Gregory patch or OCC n-sided filling).
  2. **Loop–Schaefer ACC** ("Approximating Catmull-Clark Subdivision
     Surfaces with Bicubic Patches", ACM TOG 2008 —
     [PDF](https://people.engr.tamu.edu/schaefer/research/acc.pdf),
     [DOI](https://dl.acm.org/doi/10.1145/1330511.1330519)): one bicubic
     Bézier geometry patch per face from valence-dependent masks; C⁰ across
     EV-adjacent edges with a separately constructed continuous normal
     field. Simple, closed-form, and for a *manufacturing* STEP target the
     C⁰-with-small-kink seams at EV spokes are usually within sewing
     tolerance after 1–2 refinements.
  3. Higher-order exact-G1 constructions exist (biquintic,
     [CAGD 2022](https://www.sciencedirect.com/science/article/abs/pii/S0167839622000942);
     Peters' "Patching Catmull-Clark Meshes", SIGGRAPH 2000) — more math,
     only needed if zebra-grade quality becomes a requirement.
- The limit surface is C² everywhere except **C¹ at EVs** (Peters–Reif), so
  even "perfect" output has reduced continuity at star points — same as
  Rhino and Fusion.
- **OpenSubdiv** ([opensubdiv.org](https://graphics.pixar.com/opensubdiv/))
  is the reference implementation of exactly this classification
  (`Far::PatchTable`: regular bicubic B-spline patches + Gregory end-caps).
  Modified Apache-2.0 ✅, but **no official Python binding** (`pyOpenSubdiv`
  is an unofficial `py3-none-any` wrapper of one class — not usable). Use
  it as an algorithm reference, not a dependency: the subdivision +
  patch-classification math is a few hundred lines of numpy for our needs.
- **Patents**: Alias/Autodesk SubD→NURBS conversion patents
  [US 6,950,099](https://patents.google.com/patent/US6950099) /
  [US 6,859,202](https://patents.google.com/patent/US6859202B2/en)
  (filed 2001–2002) — 20-year terms mean both are **likely expired
  (~2021–2023), verify expiry dates before shipping**. The T-splines patent
  US 7,274,364 expired 2024. The *live* patents to steer clear of are the
  Autodesk mesh-fitting family (US 10,296,664 etc., §3) covering
  rectangular-grid segmentation + constrained per-patch LSQ — the
  Stam/Catmull-Clark route does not resemble those claims.

### 6. Segmentation & fitting blocks reachable from our stack

- **CGAL Variational Shape Approximation** (Cohen-Steiner/Alliez/Desbrun
  2004; [CGAL manual](https://doc.cgal.org/latest/Surface_mesh_approximation/index.html)):
  k-proxy discrete Lloyd clustering (partition ↔ fit), hierarchical seeding,
  anchor extraction. **Findings that rule it out as a dependency:**
  (a) shipped proxies are **planes only** (L² / L²,¹ metrics) — the quadric
  proxies from the literature are *not* in CGAL ("current proxies are planes
  or vectors… generic for future extensions");
  (b) the package is **GPL** (CGAL dual GPL/commercial,
  [license page](https://www.cgal.org/license.html));
  (c) it is **not exposed in the CGAL Python bindings** (verified module
  list of [cgal-swig-bindings](https://github.com/CGAL/cgal-swig-bindings) —
  and the bindings are GPLv3+ anyway).
  **However the idea transfers**: VSA's L²,¹ (normal-based) partition-and-fit
  loop is a strict generalization of our planar region growing. A numpy
  implementation of VSA-style Lloyd iteration over our existing region
  structures (with our own quadric/B-spline proxy error) is patent-free,
  dependency-free, and would sharpen organic-region *detection* — worth
  keeping in the toolbox even though CGAL itself is unusable.
- **CGAL Shape_detection** (Schnabel-style efficient RANSAC incl. cone +
  torus; [manual](https://doc.cgal.org/latest/Shape_detection/index.html)):
  gold standard but GPL; no reliable Python exposure. Permissive
  alternatives: **pyRANSAC-3D** (Apache-2.0, plane/sphere/cylinder only,
  [repo](https://github.com/leomariga/pyRANSAC-3D)) or our existing in-house
  fitters. **No maintained permissive pip wheel of Schnabel-grade efficient
  RANSAC exists** (searched; none found — honest negative).
- **OpenCASCADE (via FreeCAD `Part` — LGPL-2.1+exception, already our
  runtime ✅):**
  - `GeomAPI_PointsToBSplineSurface` — least-squares/interpolating B-spline
    through a **rectangular grid** of points, continuity control
    ([OCC refman](https://dev.opencascade.org/doc/refman/html/class_geom_a_p_i___points_to_b_spline_surface.html)).
    The direct "sampled patch grid → NURBS face" tool. Exposed in FreeCAD as
    `Part.BSplineSurface.approximate()` / `.interpolate()`; and
    `Part.BSplineSurface` accepts explicit poles+knots — which is all that
    exact bicubic extraction needs.
  - `GeomPlate_BuildPlateSurface` / `BRepOffsetAPI_MakeFilling` — n-sided
    constrained smooth filling (docs warn "not very stable in complex
    cases") — candidate EV-cap primitive
    ([refman](https://dev.opencascade.org/doc/refman/html/class_b_rep_offset_a_p_i___make_filling.html)).
  - `ShapeUpgrade_UnifySameDomain`, sewing, `ShapeFix` — already in use.
- **FreeCAD Reverse Engineering workbench**: `ReverseEngineering.approxSurface`
  fits a NURBS to a point cloud region (configurable degree/pole grid);
  Mesh workbench has plane/cylinder/sphere segmentation. Explicitly
  early-stage, per-region, no automatic pipeline
  ([wiki](https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Reverse_Engineering_Workbench.md)) —
  a reference, not a solution.
- **gmsh** (`classifySurfaces` + `createGeometry` reparametrizes an STL into
  parametrized patches): **cannot export those discrete patches as
  STEP/BREP** — parametrization lives in the `.msh` model, only
  OpenCASCADE-factory geometry exports to STEP (verified against
  [gmsh docs](https://gmsh.info/doc/texinfo/gmsh.html) and the
  [pipeline paper](https://arxiv.org/pdf/2001.02542)). GPL-2+. pip wheels
  are excellent (macOS arm64 ✅ Windows ✅) but the export dead-end plus GPL
  make it a non-starter as our organic engine. Possible niche: out-of-process
  segmentation oracle only.
- **pymeshlab** (existing dep): isotropic explicit remeshing, screened
  Poisson, principal curvature directions — good preprocessing. **No quad
  remeshing, no VSA, no primitive fitting, no NURBS fitting** (verified
  against [filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html)).
  **License flag: pymeshlab is GPL-3.0+** and is already shipped with
  mesh2step — the redistribution posture of the existing GUI/exe bundles
  should be reviewed by the maintainer independently of this document
  (subprocess use ≠ immunity when *redistributing* the GPL wheel in a
  bundle).
- **geomdl / NURBS-Python** ([PyPI](https://pypi.org/project/geomdl/),
  [fitting docs](https://nurbs-python.readthedocs.io/en/5.x/fitting.html)):
  **MIT ✅, pure Python** (installs everywhere), global
  interpolation/least-squares surface fitting — but requires grid-organized
  input like OCC's fitter, and is slower than OCC's C++. Since FreeCAD is
  already in our runtime, OCC's fitter wins; geomdl is a fallback for
  FreeCAD-free unit tests of fitting logic.

### 7. Existing open mesh→STEP projects (prior art assessment)

| Project | Approach | Organic? | Maturity | License |
|---|---|---|---|---|
| [2STEP-Converter](https://github.com/yaneony/2STEP-Converter) | dedupe → decimate → sew triangles → repair → merge coplanar (pythonocc) | Faceted only | ~212★, v2.0.0 May 2026 | **MIT** ✅ (reference for OCC sewing tricks, incl. crash-isolated subprocesses) |
| [Stepifi](https://github.com/voron69-bit/Stepifi) | FreeCAD headless, repair + solidify | Faceted only ("does not reverse-engineer") | ~205★ | **Non-commercial** ❌ |
| [stltostp](https://github.com/slugdev/stltostp) | direct triangle STEP writer | Faceted only | stale (2019) | BSD-4-Clause ⚠️ |
| [TheTesla/stl2step](https://github.com/TheTesla/stl2step) | primitive segmentation (planes only implemented) | No | ~22★, experimental | **AGPL-3.0** ❌ |
| [Point2CAD](https://github.com/prs-eth/point2cad) (CVPR 2024, [arXiv](https://arxiv.org/pdf/2312.04962)) | per-segment analytic **or neural INR freeform** surface fit, then intersect/clip | **Yes** (strongest research freeform) | ~445★, needs ParseNet/HPNet pre-segmentation, heavy deps | **CC-BY-NC 4.0** ❌ |
| [ComplexGen](https://github.com/manycore-research/ComplexGen) (SIGGRAPH 2022) | learned corners+curves+patches → B-rep | Limited | often topologically invalid output | research **[license UNVERIFIED]** |

**Conclusion of the landscape survey: no permissive, maintained tool
converts an organic mesh to a smooth B-rep STEP.** Everything either facets,
detects primitives only, or is GPL/AGPL/NC-licensed research code. The niche
mesh2step targets is genuinely open.

---

## Part III — Recommendation

### The common denominator

All three professional architectures reduce to: **(1) get a coarse
quad-dominant control structure over the smooth region; (2) turn quads into
tensor-product spline patches; (3) special-case extraordinary vertices;
(4) knit**. Fusion does it via T-splines, Rhino via Catmull-Clark SubD,
Geomagic via an explicit patch network + constrained LSQ. The Rhino route is
the only one whose math is fully public (Catmull-Clark 1978, Stam 1998,
Loop–Schaefer 2008) and whose components exist under permissive licenses —
McNeel even confirms their evaluation "complies 100% with public domain
algorithms."

### Candidate A (recommended): Quad remesh → Catmull-Clark fit → exact bicubic B-spline extraction

*"Rhino's pipeline, in-house, tolerance-honest."*

Pipeline (per organic region or whole organic body):

1. **Preprocess** (existing): manifold3d close/union, pymeshlab decimation
   to a sane density. This is where Fusion fails and we already don't.
2. **Quad remesh** the organic region:
   - v1: `pynanoinstantmeshes` (BSD, real macOS arm64 + Windows wheels,
     numpy API) at a target quad count derived from the region's curvature
     budget.
   - v2 (quality upgrade): build **QuadriFlow** (MIT) as a small vendored
     CLI + subprocess — pure-quad, manifold, globally minimized
     singularities ⇒ fewer EVs ⇒ fewer capped patches. (Do **not** use the
     `pyQuadriFlow` wheel: `py3-none-any`, no compiled binaries, no
     license.)
   - Post-check: quad-dominance; triangulate stray non-quads into the EV
     machinery or locally re-run.
3. **Fit the control cage** (pure numpy): treat the quad mesh as a
   Catmull-Clark control cage and **least-squares-fit the cage so the limit
   surface approximates the original mesh** (limit-stencil masks are linear
   ⇒ sparse linear solve; initialize cage = quad mesh, iterate
   sample→project-to-original-mesh→re-solve 2–3 times). Without this step
   the limit surface shrinks inside the input — this is the step that makes
   the output *dimensionally honest*, and it reuses our existing
   deviation-measurement machinery as the residual metric.
4. **Subdivide 1–2×** (numpy Catmull-Clark; ~200 lines): isolates EVs so
   every face touches ≤1 extraordinary vertex.
5. **Extract patches**:
   - Regular quad face → **exact uniform bicubic B-spline patch** from its
     4×4 neighborhood (Stam) → `Part.BSplineSurface` with explicit
     poles/knots → face. Adjacent regular patches share control rows ⇒
     boundary curves match *exactly* ⇒ sewing is tolerance-trivial.
   - Optional "packed" pass later: merge regular quad grids into single
     B-spline faces (fewer faces, same geometry) — pure knot bookkeeping.
   - EV faces → **Loop–Schaefer ACC bicubic patches** (closed-form masks;
     C⁰ + small kink at EV spokes, within sew tolerance after step 4), or
     fallback `BRepOffsetAPI_MakeFilling` cap over the EV ring if ACC
     patches fail validity.
6. **Integrate with the tier ladder** (hybrid — our differentiator):
   - Region routing: after the analytic detectors claim
     planes/cylinders/cones (+fillets/spheres/tori per
     docs/CURVED_FEATURES.md), the **residual smooth chains that RTAF
     already identifies** are the organic regions. Small residuals keep the
     current faceted gap-fill; large ones (area or facet-count threshold)
     route here.
   - Whole-part shortcut: if analytic detectors claim < X% of area, skip
     analytic and run the organic pipeline on the whole body (Fusion's
     "Organic" mode equivalent, but automatic).
   - Watertight integration follows the proven **boolean-clean pattern**,
     not naive sewing: build the organic shell, then cut/fuse against the
     analytic base exactly as `build_boolean_clean_solid` does for
     cylinders — booleans recompute intersections so the organic patch
     boundary need not match analytic edges.
   - Report deviation (max/RMS to input mesh) in the existing quality
     report; expose target quad count and tolerance in `ConversionConfig`.

- **Expected quality**: smooth G2 surfaces across regular regions (exact
  bicubic lattice), C⁰/G1-approx seams only at isolated EVs — same
  qualitative structure as Rhino ToNURBS "Unpacked", the accepted industry
  result. Patch count ≈ quad count (100s–1000s, vs. 10k–100k triangles
  today).
- **Risk**: (1) EV caps failing OCC validity → mitigated by refine-once-more
  + MakeFilling fallback + per-patch revert (our `_try_boolean_step`
  discipline); (2) quad remesher robustness on junk regions → both
  remeshers are subprocess-isolated, faceted fallback always remains;
  (3) cage-fit convergence on thin features → clamp to input-mesh deviation
  budget and fall back per-region. Patent exposure: low (public-domain
  math; Alias SubD→NURBS patents likely expired — verify; the live Autodesk
  fitting patents cover the *other* architecture).
- **Dependencies**: pynanoinstantmeshes (BSD ✅, wheels ✅); optional vendored
  QuadriFlow build (MIT ✅); everything else is numpy + FreeCAD we already
  ship.
- **Effort**: the largest of the three candidates — quad remesh integration,
  CC subdivision + limit-fit solve, patch extraction, EV caps, boolean
  integration. Each stage is independently testable (pure-numpy rule
  preserved: only patch *emission* touches FreeCAD).

### Candidate B: Per-region B-spline sheet (parametrize → grid-sample → OCC fit)

*The "M4 curved-wall fallback" from CURVED_FEATURES.md, generalized.*

1. Take each residual smooth region (disk-topology after our segmentation).
2. Parametrize to 2D: harmonic/LSCM via **libigl** pip wheels (MPL2 ✅,
   macOS arm64 + Windows ✅) — or, for sweep-like bands, the region's own
   axis/arclength coordinates.
3. Resample the original mesh on a rectangular (u,v) grid;
   `GeomAPI_PointsToBSplineSurface` (via `Part.BSplineSurface.approximate`)
   → one B-spline face; trim by the projected boundary loops (existing
   boundary machinery).
4. Integrate via boolean-clean or sewing; report deviation.

- **Expected quality**: good on swept walls and gentle shells (the dominant
  M4 residual class: 60% of skipped facets); one face per region — very
  compact. Cross-region continuity is only G0 (each sheet fit
  independently) and regions with strong curvature variation or non-disk
  topology need splitting heuristics.
- **Risk**: parametrization foldover on high-curvature regions; trimming
  robustness; G0 seams read as creases. Also **stay clear of the live
  Autodesk patent family (US 10,296,664)** — that patent covers regular
  rectangular-grid segmentation of a *whole mesh* with continuity-constrained
  neighbor stitching; a per-detected-region single-patch fit without the
  constrained stitching apparatus is distinguishable, but keep it so.
- **Dependencies**: libigl (or none, for sweep-parametrized bands).
- **Effort**: smallest — it slots directly into the existing region ladder
  as the B-spline fallback the roadmap already names. **This is the right
  first milestone regardless of the long-term choice**: it attacks the
  measured residual (curved walls) and builds the sampling/fitting/trimming
  plumbing that Candidate A also needs.

### Candidate C: Patch-network auto-surface (Geomagic-style)

Explicit curvature-line extraction → quad patch network → per-patch
constrained LSQ with G1 stitching. Closest to reverse-engineering-grade
output, but: it is the hardest to automate well (the industry still keeps
humans in this loop), the continuity-constrained stitching is a large
constrained-optimization build, and it walks directly toward the **live
Autodesk patents**. Not recommended for us except as long-term inspiration
for patch-layout heuristics.

### Ranking and phasing

| Rank (implementability) | Candidate | Verdict |
|---|---|---|
| 1 | **B — per-region B-spline sheet** | Land first as M4; small, uses existing ladder; immediate RTAF payoff on curved walls |
| 2 | **A — quad remesh → CC → bicubic extraction** | The destination architecture for genuinely organic parts/whole bodies; public-domain math, MIT-clean deps, industry-validated structure |
| 3 | **C — patch-network auto-surface** | Skip: automation difficulty + live patent adjacency |

Phase plan: **B now** (M4), then **A behind a config flag** (`organic_mode`),
sharing B's deviation-report and boolean-integration plumbing. Hybrid
routing (analytic regions stay analytic; organic residual routed by
RTAF-style detection) is preserved in both — no professional tool offers
this automatically, and it remains mesh2step's differentiator.

### License / wheel compatibility summary (per dependency mentioned)

| Dependency | License | MIT-compatible? | macOS arm64 wheel | Windows wheel |
|---|---|---|---|---|
| pynanoinstantmeshes | BSD | ✅ | ✅ | ✅ |
| QuadriFlow (vendored build) | MIT | ✅ | n/a (we build) | n/a (we build) |
| libigl | MPL2 | ✅ | ✅ | ✅ |
| geomdl | MIT | ✅ | pure-py ✅ | pure-py ✅ |
| OpenSubdiv | Apache-2.0 (mod.) | ✅ (reference only) | no binding | no binding |
| FreeCAD / OCC | LGPL-2.1(+exc) | ✅ (existing runtime) | bundled | bundled |
| pyRANSAC-3D | Apache-2.0 | ✅ | pure-py ✅ | pure-py ✅ |
| pymeshlab (existing) | **GPL-3.0+** | ⚠️ review bundle redistribution | ✅ | ✅ |
| CGAL (VSA, Shape_detection, Poisson) | **GPL**/commercial | ❌ (and VSA not in Python bindings) | — | — |
| gmsh | **GPL-2+** | ❌ as engine (and can't export STEP from discrete patches) | ✅ | ✅ |
| pyQuadriFlow | **none declared** | ❌ | fake (`py3-none-any`) | fake |
| Point2CAD | **CC-BY-NC** | ❌ | — | — |
| Stepifi | non-commercial | ❌ | — | — |
| TheTesla/stl2step | **AGPL-3.0** | ❌ | — | — |
| 2STEP-Converter | MIT | ✅ (reference code) | — | — |

### Open items to verify at implementation time

- Expiry dates of Alias/Autodesk SubD→NURBS patents US 6,950,099 /
  US 6,859,202 (filed 2001–2002; expected expired, **verify**).
- pynanoinstantmeshes output guarantees (quad-dominance ratio, manifoldness)
  on our corpus — small project, needs a smoke sweep before adoption.
- Whether FreeCAD's `Part.BSplineSurface` explicit poles/knots constructor
  round-trips uniform-knot bicubic patches into STEP without re-knotting
  (expected yes; smoke-test first).
- Exact continuity order claimed by US 10,296,664 (G1 vs C²) if Candidate B
  ever grows cross-region stitching.

### Candidate A implementation notes (v0.3.0-alpha; `organic_multipatch`)

The full pipeline shipped as pure-numpy `catmull_clark.py` + `quadremesh.py`
(BSD `pynanoinstantmeshes`, provisioned optional) and OCC assembly in
`organic.py`, routed behind `organic_multipatch` by after-analytic RTAF.
Findings from building + smoke-testing it against ground truth (a coarse quad
sphere) and the target corpus (`low_poly_cat`, `3x1 Tweezer Mount`):

- **VERIFIED, works:** `pynanoinstantmeshes` imports + runs on macOS arm64
  cp311 (real wheel), emits **pure-quad, closed, manifold** cages on clean
  closed inputs (`posy=4`, `deterministic=True`; `vertex_count` is an
  edge-length target, not a hard cap — it subdivides coarse input). The numpy
  Catmull-Clark subdivision, limit-stencil projection, cage shrink-wrap fit,
  and Stam 4×4 regular-quad extraction are all correct: on an R=10 sphere the
  exact bicubic patches reproduce the radius to **max 0.20 mm (2.0 %), mean
  0.14 mm (1.4 %)** — the inherent coarse-cage limit-surface approximation.
  `buildFromPolesMultsKnots` with uniform knots `[0..7]` clamps the domain to
  the central span, so `bs.toShape()` returns exactly the per-quad patch face.
- **RESOLVED — the shell now closes watertight (v0.3.0-alpha.2).** The earlier
  blocker (a network of trimmed B-spline faces would not sew into a closed shell)
  was NOT an OCC sewing limitation of the *regular* patches — those sew flawlessly
  (2 592 contiguous edges merge, 0 free, one shell on the GT sphere) *because
  their shared boundary iso-curves are bit-identical* (verified to ~1e-15, not
  just 3 decimals). The whole gap was the **extraordinary-vertex caps**: the old
  `Part.makeFilledFace`/planar caps rebuilt their own boundary edges (which no
  longer matched the neighbour patches) and never re-sewed. The fix, using the
  `OCC.Core` (pythonocc-core) bindings **FreeCAD already ships** (verified
  importable inside FreeCAD 1.1's interpreter, `OCC 7.8.1.1`):
  1. Build each regular patch as a `Geom_BSplineSurface` +
     `BRepBuilderAPI_MakeFace(surf, u0,u1,v0,v1, tol)` (pcurves built by OCC).
     `Part.Face(surface, wire)` returning area 0 was the pcurve problem —
     `BRepBuilderAPI_MakeFace` on surface+bounds builds them.
  2. Cap each connected EV region with ONE filled face over its outer boundary
     loop, where **every boundary edge reuses the neighbour regular patch's exact
     `Geom` boundary iso-curve** (`surf.UIso/VIso` trimmed to the central span).
     The cap therefore shares the shell geometry bit-for-bit. Two fillers are
     tried — `BRepOffsetAPI_MakeFilling`, then `BRepFill_Filling` pinned by the
     region's interior limit points — each geometry-validated (bbox + local-spike
     guards) so a self-intersecting fill is rejected, plus a planar fallback.
  3. One `BRepBuilderAPI_Sewing` pass (tight tolerance, escalated to ~1–3 % of
     the cage edge length only as far as needed) → `BRepBuilderAPI_MakeSolid` →
     `ShapeFix_Shell`/`ShapeFix_Solid` healing when a C0 cap trips validity.
  Result on the ground-truth R=10 sphere: **closed watertight solid, `isValid`,
  re-reads valid from STEP, max radius deviation 0.21 mm (mean 0.08 mm), RTAF 0**
  — meeting the prior 0.20 mm accuracy. Refine-and-cap escalation (retry one
  Catmull-Clark level finer when a cap can't be validly built) closes gently
  curved organic bodies; genuinely sharp knife-edge EVs (e.g. a low-poly cat's
  ear tips, ~0.06 mm thick) still defeat all three fillers and cause that one
  region to decline (never regress).
- **Upstream cage fix (same milestone).** `pynanoinstantmeshes` leaves small
  boundary holes / non-manifold quads at some targets (the GT sphere cracks at
  target 220 but is clean at 100; `low_poly_cat` and `3x1 Tweezer Mount` cages
  failed post-repair before). `quadremesh.build_quad_cage` now (a) prefers a clean
  cage by **backing off to progressively coarser, more-robust targets**, and
  (b) **repairs** a cage as a fallback (weld coincident vertices, drop degenerate
  / non-manifold quads, fill small even boundary holes with a centroid quad fan).
  Both the tweezer and cat cages now build closed-manifold.
- **Current behaviour / corpus:** GT sphere → watertight organic solid (RTAF 0).
  `3x1 Tweezer Mount` closes watertight but is a *prismatic* part — the
  Catmull-Clark limit rounds its sharp edges to ~8.4 mm deviation, so the
  deviation gate (2 % of diag) correctly **rejects** it and it keeps its analytic
  result. `low_poly_cat` reconstructs 82/83 EV regions; the one knife-edge tip
  region defeats the fillers so it declines and keeps its faceted output. All
  three outcomes are non-regressing by the strict adoption gate (watertight +
  RTAF-improvement + bbox-stable + deviation + STEP re-read).
