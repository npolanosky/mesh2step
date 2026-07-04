/* three.js viewer for the M2SM binary mesh format.
 *
 * One reusable scene with OrbitControls. Loads an M2SM blob (see meshdata.py),
 * builds a non-indexed BufferGeometry, and toggles shaded / edges / wireframe.
 * Vertex colours (heatmap) are used when present.
 */
(function () {
  "use strict";

  const BG = 0xf0f2f5;
  const STL_COLOR = 0x8b95a1;
  const STEP_COLOR = 0x3f97cf;

  // ---- parse the M2SM binary blob into typed arrays ---------------------- //
  function parseM2SM(buffer) {
    const dv = new DataView(buffer);
    const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
    if (magic !== "M2SM") throw new Error("bad mesh magic: " + magic);
    const flags = dv.getUint32(8, true);
    const nverts = dv.getUint32(12, true);
    const hasNormals = (flags & 1) !== 0;
    const hasColors = (flags & 2) !== 0;
    let off = 16;
    const positions = new Float32Array(buffer, off, nverts * 3);
    off += nverts * 3 * 4;
    let normals = null, colors = null;
    if (hasNormals) { normals = new Float32Array(buffer, off, nverts * 3); off += nverts * 3 * 4; }
    if (hasColors) { colors = new Uint8Array(buffer, off, nverts * 3); off += nverts * 3; }
    return { positions, normals, colors, nverts };
  }

  // Weld a non-indexed triangle-soup BufferGeometry into an indexed one by
  // merging vertices at (near-)identical positions. Needed so EdgesGeometry's
  // dihedral-angle threshold can recognise coplanar neighbours and cull interior
  // edges — otherwise every triangle edge counts as a feature edge. Quantises
  // positions to a grid keyed off the geometry's bounding-box size so it's
  // scale-independent.
  function weldGeometry(geo) {
    const pos = geo.getAttribute("position");
    const n = pos.count;
    geo.computeBoundingBox();
    const bb = geo.boundingBox;
    const diag = bb.min.distanceTo(bb.max) || 1;
    const q = diag * 1e-5; // weld tolerance ~ 1e-5 of the model diagonal
    const map = new Map();
    const outPos = [];
    const index = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      const key = Math.round(x / q) + "_" + Math.round(y / q) + "_" + Math.round(z / q);
      let vi = map.get(key);
      if (vi === undefined) {
        vi = outPos.length / 3;
        map.set(key, vi);
        outPos.push(x, y, z);
      }
      index[i] = vi;
    }
    const welded = new THREE.BufferGeometry();
    welded.setAttribute("position", new THREE.BufferAttribute(new Float32Array(outPos), 3));
    welded.setIndex(index);
    welded.computeVertexNormals();
    return welded;
  }

  function Viewer(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BG);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(this.renderer.domElement);

    // Orbit mode: "constrained" (OrbitControls turntable, up-axis locked —
    // Fusion-style default) or "free" (TrackballControls, no up lock). Swapped
    // at runtime by setOrbitMode(), preserving camera position + target.
    this.orbitMode = "constrained";
    // Remember the camera's world-up so returning from free (trackball) mode —
    // which tumbles `camera.up` — can restore the turntable's up axis.
    this._up0 = this.camera.up.clone();
    this._makeControls("constrained");

    // Lighting: a key light that follows the camera + soft ambient + fill, so
    // the model reads with real shading on the light background.
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    this.key = new THREE.DirectionalLight(0xffffff, 0.75);
    this.scene.add(this.key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-1, -0.5, -1);
    this.scene.add(fill);

    this.mesh = null;
    this.edges = null;
    this.shade = "shaded";

    const self = this;
    this._resize();
    window.addEventListener("resize", function () { self._resize(); });
    (function loop() {
      requestAnimationFrame(loop);
      self.controls.update();
      self.key.position.copy(self.camera.position);
      self.renderer.render(self.scene, self.camera);
    })();
  }

  // Build the active controls object for the given mode, transferring target.
  Viewer.prototype._makeControls = function (mode) {
    const prevTarget = this.controls ? this.controls.target.clone() : null;
    if (this.controls && this.controls.dispose) this.controls.dispose();
    if (mode === "free") {
      const c = new THREE.TrackballControls(this.camera, this.renderer.domElement);
      c.rotateSpeed = 3.5;
      c.zoomSpeed = 1.2;
      c.panSpeed = 0.8;
      c.staticMoving = false;
      c.dynamicDampingFactor = 0.15;
      this.controls = c;
    } else {
      const c = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      c.enableDamping = true;
      c.dampingFactor = 0.12;
      this.controls = c;
    }
    if (prevTarget) this.controls.target.copy(prevTarget);
    this.controls.update();
  };

  // Switch orbit mode at runtime, preserving camera position + look target.
  Viewer.prototype.setOrbitMode = function (mode) {
    if (mode !== "free" && mode !== "constrained") return;
    if (mode === this.orbitMode) return;
    this.orbitMode = mode;
    // Camera position is shared (same camera object); target is transferred in
    // _makeControls. The up vector: TrackballControls tumbles it freely, so when
    // returning to constrained mode we reset up to +Z-ish is wrong for our
    // scene (Z-up parts) — restore the world up the OrbitControls expects.
    if (mode === "constrained") {
      // Free mode may have rolled camera.up; restore the original world-up so
      // the turntable behaves as it did before, without jumping the view.
      this.camera.up.copy(this._up0);
    }
    this._makeControls(mode);
  };

  Viewer.prototype._resize = function () {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.controls && this.controls.handleResize) this.controls.handleResize();
  };

  Viewer.prototype.clear = function () {
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh.material.dispose(); this.mesh = null; }
    if (this.edges) { this.scene.remove(this.edges); this.edges.geometry.dispose(); this.edges.material.dispose(); this.edges = null; }
  };

  // kind: "stl" | "step" | "heatmap"
  Viewer.prototype.load = function (buffer, kind, keepCamera) {
    this.clear();
    const parsed = parseM2SM(buffer);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
    if (parsed.normals) geo.setAttribute("normal", new THREE.BufferAttribute(parsed.normals, 3));
    else geo.computeVertexNormals();

    const opts = { flatShading: true, side: THREE.DoubleSide, roughness: 0.72, metalness: 0.02 };
    if (kind === "heatmap" && parsed.colors) {
      const f = new Float32Array(parsed.colors.length);
      for (let i = 0; i < f.length; i++) f[i] = parsed.colors[i] / 255;
      geo.setAttribute("color", new THREE.BufferAttribute(f, 3));
      opts.vertexColors = true;
    } else {
      opts.color = kind === "stl" ? STL_COLOR : STEP_COLOR;
    }
    const mat = new THREE.MeshStandardMaterial(opts);
    this.mesh = new THREE.Mesh(geo, mat);
    this.scene.add(this.mesh);

    // Edge overlay (feature edges). EdgesGeometry with a dihedral threshold only
    // culls interior edges when coplanar triangles SHARE vertices — but our
    // payload is a non-indexed triangle soup, so we first weld identical
    // positions into an indexed geometry. Without this, every triangle edge is a
    // boundary and the STEP's "Edges"/"Wireframe" looked as dense as the STL's
    // raw mesh (the reported bug). Welding makes STEP feature edges sparse.
    const welded = weldGeometry(geo);
    const egeo = new THREE.EdgesGeometry(welded, 20);
    welded.dispose();
    this.edges = new THREE.LineSegments(egeo, new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.45 }));
    this.scene.add(this.edges);

    this.setShade(this.shade);
    if (!keepCamera) this._frame(geo);
  };

  Viewer.prototype.setShade = function (mode) {
    this.shade = mode;
    if (!this.mesh) return;
    if (mode === "shaded") { this.mesh.visible = true; this.mesh.material.wireframe = false; if (this.edges) this.edges.visible = false; }
    else if (mode === "edges") { this.mesh.visible = true; this.mesh.material.wireframe = false; if (this.edges) this.edges.visible = true; }
    else if (mode === "wire") { this.mesh.visible = true; this.mesh.material.wireframe = true; if (this.edges) this.edges.visible = false; }
  };

  Viewer.prototype._frame = function (geo) {
    geo.computeBoundingSphere();
    const s = geo.boundingSphere;
    const c = s.center, r = s.radius || 1;
    const dist = r / Math.sin((this.camera.fov * Math.PI / 180) / 2) * 1.35;
    this.controls.target.copy(c);
    // Isometric-ish vantage.
    this.camera.position.set(c.x + dist * 0.7, c.y - dist * 0.7, c.z + dist * 0.6);
    this.camera.near = r / 100; this.camera.far = r * 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  };

  // Sample the centre pixel — used by automated checks to confirm a render.
  Viewer.prototype.centerPixel = function () {
    const gl = this.renderer.getContext();
    const w = this.renderer.domElement.width, h = this.renderer.domElement.height;
    const px = new Uint8Array(4);
    gl.readPixels((w / 2) | 0, (h / 2) | 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return [px[0], px[1], px[2], px[3]];
  };

  window.Viewer = Viewer;
})();
