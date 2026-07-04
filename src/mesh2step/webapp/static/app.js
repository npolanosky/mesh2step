/* mesh2step-web front-end controller. */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const api = (path, opts) => fetch(path, opts).then((r) => {
    if (!r.ok) return r.json().catch(() => ({})).then((e) => { throw new Error(e.detail || r.statusText); });
    return r;
  });

  let viewer = null;
  let selectedFiles = [];        // Files chosen but not yet converted
  let currentJob = null;         // active/last job id
  let jobDone = false;
  let evtSource = null;
  let timer = null;
  // Server-authoritative elapsed time: the SSE snapshot carries the job's
  // started/finished epochs plus the server clock ("now"); we keep the
  // client-server clock skew and tick elapsed = server_now - started. Robust
  // across page reloads and history opens — the clock never restarts.
  let clockSkew = 0;             // client_seconds - server_seconds
  let jobStarted = null;         // server epoch when the watched job started
  const meshCache = {};          // view -> ArrayBuffer for the current job

  // ---- viewer init (lazy, needs a sized container) ---------------------- //
  function ensureViewer() {
    if (!viewer) {
      viewer = new Viewer($("viewer"));
      // Apply the persisted orbit-mode choice to the freshly-built viewer.
      try {
        const m = localStorage.getItem("mesh2step.orbitMode");
        if (m === "free") viewer.setOrbitMode("free");
      } catch (e) { /* ignore */ }
    }
    return viewer;
  }

  // ---- page nav --------------------------------------------------------- //
  document.querySelectorAll(".navtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".navtab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("page-" + btn.dataset.page).classList.add("active");
      if (btn.dataset.page === "corpus") loadCorpus();
      if (btn.dataset.page === "convert" && viewer) viewer._resize();
    });
  });

  // ---- health ----------------------------------------------------------- //
  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("version").textContent = h.version || "";
    const fb = $("freecad-badge");
    if (h.freecad_ready) { fb.textContent = "FreeCAD ✓"; fb.classList.remove("pill-warn"); fb.classList.add("pill-ok"); fb.title = h.freecad; }
    else { fb.textContent = "FreeCAD ✗"; fb.title = "FreeCAD not found — conversions will fail"; }
    $("opt-savefail").checked = !!h.save_failures;
  }).catch(() => {});

  $("opt-savefail").addEventListener("change", (e) => {
    api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save_failures: e.target.checked }) });
  });

  // ---- in-progress count (main-page badge, live) ------------------------ //
  let lastActiveKey = null; // "running/queued" fingerprint to detect changes

  function refreshActive() {
    return fetch("/api/active").then((r) => r.json()).then((a) => {
      const badge = $("active-badge");
      const n = a.active || 0;
      if (n > 0) {
        let label = a.running + " running";
        if (a.queued) label += " · " + a.queued + " queued";
        badge.textContent = label;
        badge.hidden = false;
      } else {
        badge.hidden = true;
      }
      // A state transition (queued->running, running->done/cancelled) while the
      // history page is open: re-render the table so running rows' live timers
      // start/stop and final elapsed/actions appear.
      const key = a.running + "/" + a.queued;
      if (lastActiveKey !== null && key !== lastActiveKey &&
          $("page-corpus").classList.contains("active")) {
        loadCorpus();
      }
      lastActiveKey = key;
      return a;
    }).catch(() => {});
  }
  refreshActive();
  // Poll every 3s so the count stays live even for jobs started in another tab.
  setInterval(refreshActive, 3000);

  // ---- file selection --------------------------------------------------- //
  // The native OS file dialog can only be opened from a genuine user gesture.
  // Brave/macOS reopened the picker because the old dropzone-wide click handler
  // AND the <label for> both triggered input.click() for one click on "browse".
  // Fixes: (a) a dedicated Browse button is the ONLY thing that opens the dialog
  // — the dropzone body no longer does; (b) stopPropagation so the button click
  // never bubbles anywhere that could re-open it; (c) a re-entry guard that
  // ignores click requests fired while a dialog was just opened (covers the
  // focus-return / synthetic-click paths some browsers emit).
  const dz = $("dropzone");
  const fileInput = $("file-input");
  let pickerBusy = false;   // true from open until the input settles (change/focus)

  function openPicker() {
    if (pickerBusy) return;         // guard re-entry (double-fire)
    pickerBusy = true;
    fileInput.value = "";           // allow re-selecting the same file
    fileInput.click();
    // Release the guard once focus returns to the window (dialog closed) or a
    // change lands, whichever first. A timed fallback covers a cancelled dialog
    // that emits neither on some platforms.
    const release = () => { pickerBusy = false; window.removeEventListener("focus", release); };
    window.addEventListener("focus", release, { once: true });
    setTimeout(() => { pickerBusy = false; }, 1500);
  }

  fileInput.addEventListener("change", (e) => {
    pickerBusy = false;
    if (e.target.files.length) addFiles(e.target.files);
  });

  $("browse-btn").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();            // don't let it bubble to any dropzone handler
    openPicker();
  });

  // Drag & drop still works on the whole dropzone; clicking the dropzone body
  // deliberately does NOT open the dialog (only the Browse button does).
  ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });

  // Multiple files may be selected/dropped at once; each becomes its own job
  // on Convert. Non-.stl entries are dropped with a note.
  function addFiles(list) {
    const files = [...list].filter((f) => f.name.toLowerCase().endsWith(".stl"));
    if (!files.length) { alert("Please choose .stl files."); return; }
    if (files.length < list.length) alert("Skipped " + (list.length - files.length) + " non-.stl file(s).");
    for (const f of files) {
      // De-dupe by name+size so re-dropping the same selection doesn't stack.
      if (!selectedFiles.some((s) => s.name === f.name && s.size === f.size)) selectedFiles.push(f);
    }
    renderFileList();
    // Reset any prior result view and preview the first file client-side.
    resetResult();
    previewLocalStl(selectedFiles[0]);
  }

  function renderFileList() {
    const el = $("file-list");
    el.innerHTML = "";
    el.hidden = selectedFiles.length === 0;
    selectedFiles.forEach((f, i) => {
      const row = document.createElement("div");
      row.className = "file-row";
      const name = document.createElement("span");
      name.className = "file-row-name";
      name.textContent = f.name;
      const size = document.createElement("span");
      size.className = "file-row-size";
      size.textContent = (f.size / 1024).toFixed(0) + " KB";
      const rm = document.createElement("button");
      rm.className = "file-row-rm";
      rm.title = "Remove";
      rm.textContent = "✕";
      rm.addEventListener("click", () => {
        selectedFiles.splice(i, 1);
        renderFileList();
        if (selectedFiles.length) previewLocalStl(selectedFiles[0]);
      });
      row.appendChild(name); row.appendChild(size); row.appendChild(rm);
      el.appendChild(row);
    });
    const btn = $("convert-btn");
    btn.disabled = selectedFiles.length === 0;
    btn.textContent = selectedFiles.length > 1
      ? "Convert " + selectedFiles.length + " files → STEP" : "Convert → STEP";
  }

  // Parse a binary/ascii STL in the browser for instant input preview.
  function previewLocalStl(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const blob = stlToM2SM(reader.result);
        $("viewer-empty").hidden = true;
        ensureViewer().load(blob, "stl", false);
        setViewTab("stl");
        viewer._resize();
      } catch (err) { console.warn("local STL preview failed:", err); }
    };
    reader.readAsArrayBuffer(file);
  }

  // Minimal client-side STL -> M2SM (positions + face normals), binary or ascii.
  function stlToM2SM(buffer) {
    const bytes = new Uint8Array(buffer);
    let tris;
    const head = String.fromCharCode.apply(null, bytes.slice(0, 5)).toLowerCase();
    const dv = new DataView(buffer);
    const isBinaryByCount = buffer.byteLength >= 84 && (84 + 50 * dv.getUint32(80, true) === buffer.byteLength);
    if (head === "solid" && !isBinaryByCount) {
      tris = parseAsciiStl(new TextDecoder("latin1").decode(bytes));
    } else {
      tris = parseBinaryStl(dv);
    }
    return buildM2SM(tris);
  }
  function parseBinaryStl(dv) {
    const n = dv.getUint32(80, true);
    const tris = new Float32Array(n * 9);
    let off = 84, o = 0;
    for (let i = 0; i < n; i++) {
      off += 12; // skip normal
      for (let v = 0; v < 9; v++) { tris[o++] = dv.getFloat32(off, true); off += 4; }
      off += 2; // attr
    }
    return tris;
  }
  function parseAsciiStl(text) {
    const nums = [];
    const re = /vertex\s+(\S+)\s+(\S+)\s+(\S+)/g; let m;
    while ((m = re.exec(text))) { nums.push(+m[1], +m[2], +m[3]); }
    return new Float32Array(nums);
  }
  function buildM2SM(pos) {
    const nverts = pos.length / 3;
    const normals = new Float32Array(pos.length);
    for (let i = 0; i < nverts; i += 3) {
      const ax = pos[i*3], ay = pos[i*3+1], az = pos[i*3+2];
      const bx = pos[i*3+3], by = pos[i*3+4], bz = pos[i*3+5];
      const cx = pos[i*3+6], cy = pos[i*3+7], cz = pos[i*3+8];
      let nx = (by-ay)*(cz-az)-(bz-az)*(cy-ay);
      let ny = (bz-az)*(cx-ax)-(bx-ax)*(cz-az);
      let nz = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax);
      const len = Math.hypot(nx, ny, nz) || 1; nx/=len; ny/=len; nz/=len;
      for (let k = 0; k < 3; k++) { normals[(i+k)*3]=nx; normals[(i+k)*3+1]=ny; normals[(i+k)*3+2]=nz; }
    }
    const header = new ArrayBuffer(16);
    const hv = new DataView(header);
    hv.setUint8(0,77); hv.setUint8(1,50); hv.setUint8(2,83); hv.setUint8(3,77); // "M2SM"
    hv.setUint32(4,1,true); hv.setUint32(8,1,true); hv.setUint32(12,nverts,true); // flags=normals
    const out = new Uint8Array(16 + pos.byteLength + normals.byteLength);
    out.set(new Uint8Array(header), 0);
    out.set(new Uint8Array(pos.buffer, pos.byteOffset, pos.byteLength), 16);
    out.set(new Uint8Array(normals.buffer), 16 + pos.byteLength);
    return out.buffer;
  }

  // ---- convert ---------------------------------------------------------- //
  $("convert-btn").addEventListener("click", () => {
    if (!selectedFiles.length) return;
    const options = {
      source_units: $("units").value,
      detect_cylinders: $("opt-detect").checked,
      repair_mesh: $("opt-repair").checked,
      full_closed: $("opt-closed").checked,
      faceted: $("opt-faceted").checked,
    };
    const files = selectedFiles.slice();
    $("convert-btn").disabled = true;
    resetResult();
    $("progress-card").hidden = false;
    setStatus(files.length > 1 ? "Uploading " + files.length + " files…" : "Uploading…", 2);

    // Submit every file as its own job (one POST each) with the same options;
    // the server queues them in order. We then watch the FIRST job live — the
    // rest stay reachable from History / the active badge.
    (async () => {
      const ids = [];
      for (const f of files) {
        const fd = new FormData();
        fd.append("file", f, f.name);
        fd.append("options", JSON.stringify(options));
        const r = await api("/api/convert", { method: "POST", body: fd });
        ids.push((await r.json()).id);
      }
      return ids;
    })().then((ids) => {
      selectedFiles = [];
      renderFileList();
      refreshActive();
      watchJob(ids[0]);
      if (ids.length > 1) {
        appendLog(ids.length + " jobs queued — watching the first; the others run "
          + "in order (open them from Corpus & history).", "l-stage");
      }
    }).catch((err) => { setStatus("Failed: " + err.message, 0); renderFileList(); showCancel(false); });
  });

  // ---- cancel ----------------------------------------------------------- //
  function showCancel(on) {
    const b = $("cancel-btn");
    b.hidden = !on;
    b.disabled = false;
  }
  function cancelJob(id, btn) {
    if (!id) return Promise.resolve();
    if (btn) btn.disabled = true;
    return api("/api/jobs/" + id + "/cancel", { method: "POST" })
      .then((r) => r.json())
      .catch((err) => { if (btn) btn.disabled = false; appendLog("cancel failed: " + err.message, "l-err"); throw err; });
  }
  $("cancel-btn").addEventListener("click", () => {
    cancelJob(currentJob, $("cancel-btn")).then(() => setStatus("Cancelling…", null));
  });

  // ---- new conversion (reset without refresh) ---------------------------- //
  // Returns the page to a fresh upload state. The job we were watching keeps
  // running server-side — only our SSE subscription is dropped; its progress
  // stays reachable from History (Open) and the active badge.
  function newConversion() {
    if (evtSource) { evtSource.close(); evtSource = null; }
    clearInterval(timer);
    currentJob = null;
    jobDone = false;
    jobStarted = null;
    selectedFiles = [];
    renderFileList();
    resetResult();
    $("progress-card").hidden = true;
    showCancel(false);
    setStatus("Ready", 0);
    $("elapsed").textContent = "0.0s";
    if (viewer) viewer.clear();
    const ve = $("viewer-empty");
    ve.textContent = "Select an STL to preview it here.";
    ve.hidden = false;
    setViewTab("stl");
    refreshActive();
  }
  $("new-conv-btn").addEventListener("click", newConversion);

  function resetResult() {
    jobDone = false;
    for (const k in meshCache) delete meshCache[k];
    $("verdict").hidden = true;
    $("result-badges").hidden = true;
    $("result-actions").hidden = true;
    $("log").textContent = "";
    $("scalebar").hidden = true;
    $("dev-stats").hidden = true;
    $("view-tabs").querySelectorAll(".seg-btn").forEach((b) => { if (b.dataset.view !== "stl") b.disabled = true; });
  }

  function setStatus(text, pct) {
    $("status-line").textContent = text;
    if (pct != null) $("progress-bar").style.width = pct + "%";
  }
  function appendLog(line, cls) {
    const el = $("log");
    const span = document.createElement("span");
    span.className = cls || "";
    span.textContent = line + "\n";
    el.appendChild(span); el.scrollTop = el.scrollHeight;
  }

  const _TERMINAL = { done: 1, failed: 1, cancelled: 1 };

  function watchJob(id) {
    currentJob = id;
    jobDone = false;
    jobStarted = null;
    clearInterval(timer);
    // The clock is server-authoritative: elapsed = server_now - started (from
    // the SSE snapshot / running event), NOT time-since-page-load — so a reload
    // or a history Open mid-run shows the true process time.
    timer = setInterval(() => {
      if (jobDone) return;
      if (jobStarted != null) {
        const nowSrv = Date.now() / 1000 - clockSkew;
        $("elapsed").textContent = Math.max(0, nowSrv - jobStarted).toFixed(1) + "s";
      } else {
        $("elapsed").textContent = "queued";
      }
    }, 100);
    if (evtSource) evtSource.close();
    showCancel(true);
    evtSource = new EventSource("/api/jobs/" + id + "/events");
    evtSource.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "snapshot") {
        if (ev.now) clockSkew = Date.now() / 1000 - ev.now;
        if (ev.started) jobStarted = ev.started;
        setStatus(ev.status || "Working…", ev.progress || 2);
        (ev.log || []).forEach((l) => appendLog(l.startsWith("PROGRESS:") ? l.slice(9).trim() : l,
          l.startsWith("PROGRESS:") ? "l-stage" : ""));
        // An already-terminal job (e.g. Open on a just-finished row) still needs
        // its result rendered; the server also sends a follow-up state event.
      } else if (ev.type === "progress") {
        setStatus(ev.message, ev.progress);
        appendLog(ev.message, "l-stage");
      } else if (ev.type === "log") {
        appendLog(ev.message, /error|traceback/i.test(ev.message) ? "l-err" : "");
      } else if (ev.type === "state" && ev.state === "running") {
        // A watched queued job just started: base the clock on its true start.
        if (ev.started) jobStarted = ev.started;
      } else if (ev.type === "state" && _TERMINAL[ev.state]) {
        evtSource.close();
        finishJob(id, ev.state, ev.error);
      }
    };
    evtSource.onerror = () => { /* SSE auto-reconnects; ignore transient drops */ };
  }

  // Re-attach the convert page to an existing job (from the history "Open"
  // button): reset the UI, switch to the convert page, replay its live SSE +
  // state so the user watches a job started elsewhere as if they'd started it.
  function attachToJob(id) {
    // Switch to the convert page.
    document.querySelectorAll(".navtab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    document.querySelector('.navtab[data-page="convert"]').classList.add("active");
    $("page-convert").classList.add("active");
    if (viewer) viewer._resize();

    selectedFiles = [];
    renderFileList();
    resetResult();
    $("progress-card").hidden = false;
    $("convert-btn").disabled = true;
    setStatus("Attaching…", 2);
    // Preview the job's input STL in the viewer.
    $("viewer-empty").hidden = true;
    fetch("/api/jobs/" + id + "/mesh/stl").then((r) => r.arrayBuffer()).then((buf) => {
      meshCache["stl"] = { buf, stats: null };
      ensureViewer().load(buf, "stl", false); setViewTab("stl"); viewer._resize();
    }).catch(() => {});
    watchJob(id);
  }

  function finishJob(id, state, error) {
    jobDone = true;
    showCancel(false);
    $("convert-btn").disabled = !selectedFiles.length;
    refreshActive();
    fetch("/api/jobs/" + id).then((r) => r.json()).then((job) => {
      // Final elapsed comes from the server (finished - started), never the
      // page-load clock.
      if (job.elapsed) $("elapsed").textContent = job.elapsed.toFixed(1) + "s";
      renderResult(job, state, error);
    });
  }

  function renderResult(job, state, error) {
    const v = $("verdict");
    if (state === "cancelled") {
      v.className = "verdict warnings"; v.textContent = "⊘ Cancelled";
      v.hidden = false;
      appendLog(error || "cancelled", "l-err");
      setStatus("Cancelled", 0);
      return;
    }
    if (state === "failed") {
      v.className = "verdict problems"; v.textContent = "✖ Conversion failed";
      v.hidden = false;
      appendLog(error || "unknown error", "l-err");
      setStatus("Failed", 100);
      return;
    }
    const s = (job.result && job.result.stats) || {};
    const quality = s.quality || "good";
    const label = { good: "✔ GOOD", warnings: "⚠ OK — with warnings", problems: "✖ PROBLEMS" }[quality] || "done";
    v.className = "verdict " + quality; v.textContent = label; v.hidden = false;
    setStatus("Done → " + (job.outputs[0] || "step"), 100);

    // Badges: watertight, valid, method, faces, RTAF, timing.
    const badges = [];
    badges.push(bdg(s.is_solid ? "watertight ✓" : "not watertight", s.is_solid ? "b-ok" : "b-err"));
    if (job.result && job.result.method) badges.push(bdg(job.result.method, ""));
    if (s.faces_out != null) badges.push(bdg("faces " + (s.faces_in||"?") + "→" + s.faces_out, ""));
    if (s.rtaf != null) badges.push(bdg("RTAF " + (s.rtaf*100).toFixed(0) + "%", s.rtaf >= 0.05 ? "b-warn" : "b-ok"));
    if (job.elapsed) badges.push(bdg(job.elapsed.toFixed(1) + "s", ""));
    const bc = $("result-badges"); bc.innerHTML = ""; badges.forEach((b) => bc.appendChild(b)); bc.hidden = false;

    for (const w of (s.warnings || [])) appendLog("⚠ " + w, "l-err");

    // Result actions: one download link PER output (dual-output jobs offer
    // both files), every href keyed by this job's id — never in-memory state.
    $("result-actions").hidden = false;
    const dlc = $("download-links");
    dlc.innerHTML = "";
    const outs = job.outputs && job.outputs.length ? job.outputs : [];
    outs.forEach((name) => {
      const a = document.createElement("a");
      a.className = "btn-secondary";
      a.href = "/api/jobs/" + job.id + "/download?name=" + encodeURIComponent(name);
      a.setAttribute("download", name);
      a.textContent = outs.length > 1 ? "Download " + name : "Download STEP";
      a.title = name;
      dlc.appendChild(a);
    });
    $("flag-btn").disabled = !s.is_solid;
    $("flag-btn").onclick = () => {
      $("flag-btn").disabled = true;
      api("/api/jobs/" + job.id + "/flag", { method: "POST" })
        .then(() => appendLog("Flagged for improvement (faceted_improvable).", "l-ok"))
        .catch((e) => appendLog("flag failed: " + e.message, "l-err"));
    };

    // Enable STEP + heatmap viewer tabs.
    $("view-tabs").querySelectorAll(".seg-btn").forEach((b) => b.disabled = false);
    loadView("step");  // auto-jump to the converted result
  }
  function bdg(text, cls) { const s = document.createElement("span"); s.className = "badge " + (cls||""); s.textContent = text; return s; }

  // ---- viewer tabs ------------------------------------------------------ //
  $("view-tabs").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b || b.disabled) return;
    loadView(b.dataset.view);
  });
  $("shade-tabs").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    $("shade-tabs").querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    if (viewer) viewer.setShade(b.dataset.shade);
  });

  // ---- orbit mode (constrained turntable vs free trackball) ------------- //
  const ORBIT_KEY = "mesh2step.orbitMode";
  function applyOrbitMode(mode) {
    $("orbit-tabs").querySelectorAll(".seg-btn").forEach((x) =>
      x.classList.toggle("active", x.dataset.orbit === mode));
    if (viewer) viewer.setOrbitMode(mode);
    try { localStorage.setItem(ORBIT_KEY, mode); } catch (e) { /* private mode */ }
  }
  $("orbit-tabs").addEventListener("click", (e) => {
    const b = e.target.closest(".seg-btn"); if (!b) return;
    applyOrbitMode(b.dataset.orbit);
  });
  // Restore the persisted choice (applied to the viewer lazily once it exists).
  let savedOrbit = "constrained";
  try { savedOrbit = localStorage.getItem(ORBIT_KEY) || "constrained"; } catch (e) { /* ignore */ }
  $("orbit-tabs").querySelectorAll(".seg-btn").forEach((x) =>
    x.classList.toggle("active", x.dataset.orbit === savedOrbit));
  function setViewTab(view) {
    $("view-tabs").querySelectorAll(".seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  }

  function viewerMessage(text) {
    // Surface a viewer-load problem where the user can actually see it, instead
    // of only in the collapsed log (the old silent-blank behaviour).
    const el = $("viewer-empty");
    el.textContent = text;
    el.hidden = false;
  }

  function loadView(view) {
    setViewTab(view);
    $("scalebar").hidden = view !== "heatmap";
    $("dev-stats").hidden = view !== "heatmap";
    if (view === "stl" && selectedFiles.length && !currentJob) return; // already local-previewed
    const url = "/api/jobs/" + currentJob + "/mesh/" + view;
    if (meshCache[view]) { renderMesh(meshCache[view].buf, view, meshCache[view].stats); return; }
    $("viewer-empty").hidden = true;
    if (view === "heatmap") viewerMessage("Computing deviation heatmap…"), $("viewer-empty").hidden = false;
    fetch(url).then((r) => {
      // A non-2xx response is a JSON error body, NOT an M2SM blob. Parse it and
      // surface the detail rather than feeding error bytes to the mesh parser
      // (which used to fail with a silent "bad mesh magic").
      if (!r.ok) {
        return r.json().catch(() => ({})).then((e) => {
          throw new Error(e.detail || ("HTTP " + r.status));
        });
      }
      const stats = r.headers.get("X-Deviation-Stats");
      return r.arrayBuffer().then((buf) => ({ buf, stats: stats ? JSON.parse(stats) : null }));
    }).then(({ buf, stats }) => {
      meshCache[view] = { buf, stats };
      renderMesh(buf, view, stats);
      $("viewer-empty").hidden = true;
    }).catch((err) => {
      const label = view === "heatmap" ? "Deviation heatmap unavailable" :
                    view === "step" ? "STEP preview unavailable" : "Preview unavailable";
      appendLog("viewer load failed (" + view + "): " + err.message, "l-err");
      viewerMessage(label + ": " + err.message);
    });
  }

  function renderMesh(buf, view, stats) {
    ensureViewer().load(buf, view, false);
    viewer._resize();
    if (view === "heatmap" && stats) {
      $("sb-hi").textContent = stats.clamp.toFixed(3);
      $("dev-stats").innerHTML =
        "max " + stats.max.toFixed(4) + " mm<br>" +
        "rms " + stats.rms.toFixed(4) + " mm<br>" +
        "p95 " + stats.p95.toFixed(4) + " mm";
      $("dev-stats").hidden = false;
    }
  }

  // ---- corpus + history ------------------------------------------------- //
  let historySkew = 0; // client_seconds - server_seconds, from /api/jobs "now"

  function loadCorpus() {
    fetch("/api/jobs").then((r) => r.json()).then((d) => {
      if (d.now) historySkew = Date.now() / 1000 - d.now;
      renderHistory(d.jobs);
    });
    fetch("/api/corpus").then((r) => r.json()).then((d) => renderCorpus(d));
  }

  // Tick running rows' Elapsed in place (no table re-render): each running row
  // renders a span carrying its server started-epoch; one interval updates the
  // text. Terminal transitions re-render the table via the active poll below,
  // which replaces the live spans with the final server-computed elapsed.
  setInterval(() => {
    document.querySelectorAll("#history .elapsed-live").forEach((el) => {
      const st = parseFloat(el.dataset.started);
      if (!isNaN(st)) {
        const nowSrv = Date.now() / 1000 - historySkew;
        el.textContent = Math.max(0, nowSrv - st).toFixed(1) + "s";
      }
    });
  }, 500);

  function renderHistory(jobs) {
    const el = $("history");
    if (!jobs.length) { el.innerHTML = '<p class="muted">No conversions yet.</p>'; return; }
    let h = "<table><thead><tr><th>File</th><th>State</th><th>Quality</th><th>Watertight</th><th>Method</th><th>Elapsed</th><th></th></tr></thead><tbody>";
    for (const j of jobs) {
      const s = (j.result && j.result.stats) || {};
      const active = j.state === "running" || j.state === "queued";
      let actions = "<div class='row-actions'>";
      if (active) {
        // Running/queued jobs: attach to their live progress, or stop them.
        if (j.state === "running") actions += "<button class='btn-mini open' data-open='" + j.id + "'>Open</button>";
        actions += "<button class='btn-mini stop' data-stop='" + j.id + "'>Stop</button>";
      } else {
        // Terminal jobs (done/failed/cancelled): Open loads the full result
        // (verdict, badges, viewer tabs, downloads) into the convert page —
        // everything is keyed by this row's job id, not any "current" job.
        actions += "<button class='btn-mini open' data-open='" + j.id + "'>Open</button>";
        if (j.state === "done") {
          const outs = j.outputs || [];
          outs.forEach((name) => {
            actions += "<a class='btn-mini' href='/api/jobs/" + j.id +
              "/download?name=" + encodeURIComponent(name) + "' download='" + esc(name) +
              "' title='" + esc(name) + "'>⤓" + (outs.length > 1 ? " " + esc(name) : "") + "</a>";
          });
        }
        actions += "<button class='btn-mini' data-rerun='" + j.id + "'>Re-run</button>";
      }
      actions += "</div>";
      // Running rows tick live from the server started-epoch; queued rows show
      // a dash; terminal rows show the final server-computed elapsed.
      const elapsedCell = (j.state === "running" && j.started)
        ? "<span class='elapsed-live' data-started='" + j.started + "'>…</span>"
        : (j.elapsed ? j.elapsed.toFixed(1) + "s" : "—");
      h += "<tr><td class='wrap'>" + esc(j.filename) + "</td>" +
        "<td><span class='tag " + j.state + "'>" + j.state + "</span></td>" +
        "<td>" + (s.quality || "—") + "</td>" +
        "<td>" + (s.is_solid == null ? "—" : (s.is_solid ? "✓" : "✗")) + "</td>" +
        "<td>" + ((j.result && j.result.method) || "—") + "</td>" +
        "<td>" + elapsedCell + "</td>" +
        "<td>" + actions + "</td></tr>";
    }
    el.innerHTML = h + "</tbody></table>";
    el.querySelectorAll("[data-rerun]").forEach((b) => b.addEventListener("click", () => {
      api("/api/jobs/" + b.dataset.rerun + "/rerun", { method: "POST" })
        .then((r) => r.json()).then((d) => { attachToJob(d.id); refreshActive(); });
    }));
    el.querySelectorAll("[data-open]").forEach((b) => b.addEventListener("click", () => {
      attachToJob(b.dataset.open);
    }));
    el.querySelectorAll("[data-stop]").forEach((b) => b.addEventListener("click", () => {
      cancelJob(b.dataset.stop, b).then(() => { setTimeout(loadCorpus, 300); refreshActive(); }).catch(() => {});
    }));
  }
  function renderCorpus(d) {
    $("corpus-dest").textContent = "→ " + d.dest;
    const el = $("corpus");
    if (!d.files.length) { el.innerHTML = '<p class="muted">Corpus is empty.</p>'; return; }
    let h = "<table><thead><tr><th>File</th><th>Category</th><th>Quality</th><th>RTAF</th><th>First seen</th><th class='wrap'>Reason</th></tr></thead><tbody>";
    for (const f of d.files) {
      h += "<tr><td class='wrap'>" + esc(f.original_name || f.file) + "</td>" +
        "<td><span class='tag'>" + esc(f.category || "?") + "</span></td>" +
        "<td>" + (f.quality || "—") + "</td>" +
        "<td>" + (f.rtaf != null ? (f.rtaf*100).toFixed(0) + "%" : "—") + "</td>" +
        "<td>" + esc((f.first_seen || "").slice(0, 10)) + "</td>" +
        "<td class='wrap muted'>" + esc(f.error || "") + "</td></tr>";
    }
    el.innerHTML = h + "</tbody></table>";
  }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
})();
