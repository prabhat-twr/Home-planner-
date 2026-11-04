(() => {
  "use strict";

  // ---------------------------------------------------------------- config
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const qsApi = new URLSearchParams(location.search).get("api");
  if (qsApi) store.set("soulhome.api", qsApi);
  const API = (qsApi || store.get("soulhome.api") || window.SOULHOME_API || "").replace(/\/$/, "");

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ELEMENT = { fire: "var(--fire)", water: "var(--water)", earth: "var(--earth)", air: "var(--air)", space: "var(--space)" };
  const DIRS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];

  const STAGE_LABEL = {
    load_inputs: "Read upload", layout_inference: "Room layout (HorizonNet)", depth: "Depth estimation",
    layout: "Walls & ceiling", openings: "Doors & windows", detect: "Object detection (YOLO11)",
    place_objects: "Place objects in 3D", normalize_units: "Units & up-axis", mesh_cleanup: "Mesh cleanup",
    decimate: "Decimation", sample_points: "Surface sampling", downsample: "Voxel downsampling",
    gravity_align: "Gravity alignment", floorplan: "Floor plan, doors & windows", export_mesh: "Export clean mesh",
    assemble: "Build spatial model", validate: "Geometry validation", write_artifacts: "Render floor plan",
  };
  const PLAN = {
    "panorama-photo": ["load_inputs", "layout_inference", "depth", "layout", "openings", "detect", "place_objects", "assemble", "validate", "write_artifacts"],
    panorama: ["load_inputs", "depth", "layout", "openings", "place_objects", "assemble", "validate", "write_artifacts"],
    lidar: ["load_inputs", "normalize_units", "mesh_cleanup", "decimate", "sample_points", "gravity_align", "floorplan", "place_objects", "export_mesh", "assemble", "validate", "write_artifacts"],
    arcore: ["load_inputs", "normalize_units", "downsample", "gravity_align", "floorplan", "place_objects", "assemble", "validate", "write_artifacts"],
  };

  // ------------------------------------------------------------------- api
  async function api(path, opts = {}) {
    const r = await fetch(API + path, opts);
    const ct = r.headers.get("content-type") || "";
    const body = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) throw new Error((body && body.detail) || `${r.status} ${r.statusText}`);
    return body;
  }

  async function health() {
    const el = $("#apiStatus");
    try {
      const h = await api("/healthz");
      el.className = "status ok";
      el.lastElementChild.textContent = `API online · pipeline v${h.pipeline_version}`;
    } catch {
      el.className = "status bad";
      el.lastElementChild.textContent = "API unreachable";
    }
  }

  // ------------------------------------------------------------------ tabs
  $$("[data-tab]").forEach((t) => t.addEventListener("click", () => {
    $$("[data-tab]").forEach((x) => x.classList.toggle("active", x === t));
    $("#pane-demo").classList.toggle("hidden", t.dataset.tab !== "demo");
    $("#pane-upload").classList.toggle("hidden", t.dataset.tab !== "upload");
  }));
  $$("[data-rtab]").forEach((t) => t.addEventListener("click", () => showRTab(t.dataset.rtab)));
  function showRTab(name) {
    $$("[data-rtab]").forEach((x) => x.classList.toggle("active", x.dataset.rtab === name));
    $$(".rpane").forEach((p) => p.classList.toggle("hidden", p.id !== "r-" + name));
  }

  // ----------------------------------------------------------------- upload
  let tier = "panorama";
  const HINTS = {
    panorama: "Equirectangular JPG/PNG (2:1)",
    lidar: "USDZ, GLB, PLY or OBJ mesh (+ optional detections.json)",
    arcore: "PLY point cloud or mesh (+ optional detections.json)",
  };
  $$("#tierSeg button").forEach((b) => b.addEventListener("click", () => {
    tier = b.dataset.tier;
    $$("#tierSeg button").forEach((x) => x.classList.toggle("on", x === b));
    $("#dropHint").textContent = HINTS[tier];
    $("#heightWrap").classList.toggle("hidden", tier !== "panorama");
  }));
  const drop = $("#drop"), fileInput = $("#files");
  const showFiles = () => {
    const f = [...fileInput.files];
    $("#dropText").innerHTML = f.length ? `<b>${f.map((x) => esc(x.name)).join(", ")}</b><br><small>${(f.reduce((a, x) => a + x.size, 0) / 1e6).toFixed(1)} MB</small>`
      : `<b>Drop files here</b> or click to choose<br><small id="dropHint">${HINTS[tier]}</small>`;
  };
  fileInput.addEventListener("change", showFiles);
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => { fileInput.files = ev.dataTransfer.files; showFiles(); });

  $("#pane-upload").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!fileInput.files.length) { alert("Choose at least one file."); return; }
    const meta = { room_name: $("#roomName").value || "room", north_deg: Number($("#north").value) || 0 };
    if (tier === "panorama") meta.camera_height_m = Number($("#height").value) || 1.5;
    const fd = new FormData();
    fd.append("tier", tier);
    fd.append("metadata", JSON.stringify(meta));
    [...fileInput.files].forEach((f) => fd.append("files", f, f.name));
    const key = (crypto.randomUUID && crypto.randomUUID()) || String(Date.now() + Math.random());
    start(tier, () => api("/v1/scans", { method: "POST", body: fd, headers: { "Idempotency-Key": key } }));
  });

  $$("[data-demo]").forEach((b) => b.addEventListener("click", () =>
    start(b.dataset.demo, () => api(`/v1/demo/${b.dataset.demo}`, { method: "POST" }))));

  // ------------------------------------------------------------ job flow
  let pollTimer = null, activeId = null;

  async function start(kind, submit) {
    setBusy(true);
    showProgress(kind, null);
    try {
      const job = await submit();
      track(job.id, kind);
    } catch (e) {
      showError(e.message);
      setBusy(false);
    }
  }

  function setBusy(b) {
    $$("[data-demo]").forEach((x) => (x.disabled = b));
    $("#uploadBtn").disabled = b;
  }

  function planFor(kind, job) {
    if (PLAN[kind]) return PLAN[kind];
    if (job && job.tier === "panorama") return job.files && job.files.includes("depth.npy") ? PLAN.panorama : PLAN["panorama-photo"];
    return PLAN[(job && job.tier) || "lidar"];
  }

  function showProgress(kind, job) {
    $("#empty").classList.add("hidden");
    $("#result").classList.add("hidden");
    $("#progress").classList.remove("hidden");
    $("#err").classList.add("hidden");
    $("#progTitle").textContent = "Processing…";
    $("#progId").textContent = job ? job.id : "";
    renderStages(planFor(kind, job), null, job);
  }

  function renderStages(plan, current, job) {
    const done = new Set((job && job.summary && job.summary.stages || []).map((s) => s.name));
    let idx = current ? plan.indexOf(current) : -1;
    if (current && idx < 0) { plan = [...plan, current]; idx = plan.length - 1; }
    const status = job ? job.status : "queued";
    $("#stages").innerHTML = (status === "queued" ? [`<li class="now"><span class="ic"></span>Waiting for a worker${kindNote(job)}</li>`] : [])
      .concat(plan.map((s, i) => {
        const cls = done.has(s) || (idx >= 0 && i < idx) ? "done" : i === idx ? "now" : "";
        return `<li class="${cls}"><span class="ic"></span>${esc(STAGE_LABEL[s] || s)}</li>`;
      })).join("");
  }
  const kindNote = (job) => (job && job.tier === "panorama" && !(job.files || []).includes("depth.npy") ? " (GPU may be starting up)" : "");

  function track(id, kind) {
    clearTimeout(pollTimer);
    activeId = id;
    const tick = async () => {
      if (activeId !== id) return;
      let job;
      try { job = await api(`/v1/scans/${id}`); } catch (e) { pollTimer = setTimeout(tick, 4000); return; }
      $("#progId").textContent = job.id;
      if (job.status === "succeeded") { setBusy(false); await showResult(job); loadHistory(); return; }
      if (job.status === "failed") {
        setBusy(false);
        renderStages(planFor(kind, job), job.error && job.error.stage, job);
        showError(`${job.error.stage} · ${job.error.code}: ${job.error.message}`);
        loadHistory();
        return;
      }
      $("#progTitle").textContent = job.status === "retrying" ? `Retrying (attempt ${job.attempts + 1})…` : "Processing…";
      renderStages(planFor(kind, job), job.current_stage, job);
      pollTimer = setTimeout(tick, (job.poll_after_s || 2) * 1000);
    };
    tick();
  }

  function showError(msg) {
    $("#progress").classList.remove("hidden");
    $("#progTitle").textContent = "Scan failed";
    const e = $("#err");
    e.textContent = msg;
    e.classList.remove("hidden");
  }

  // ---------------------------------------------------------------- result
  async function showResult(job) {
    const [model, svg] = await Promise.all([api(`/v1/scans/${job.id}/model`), api(`/v1/scans/${job.id}/floorplan.svg`)]);
    const room = model.rooms[0];
    const s = job.summary;
    $("#progress").classList.add("hidden");
    $("#empty").classList.add("hidden");
    $("#result").classList.remove("hidden");

    const kpi = (v, l) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`;
    $("#kpis").innerHTML = [
      kpi(s.floor_area_m2.toFixed(1), "m² floor area"), kpi(s.ceiling_height_m.toFixed(2), "m ceiling"),
      kpi(s.walls, "walls"), kpi(s.doors, s.doors === 1 ? "door" : "doors"), kpi(s.windows, s.windows === 1 ? "window" : "windows"),
      kpi(s.objects, "objects"), kpi(Math.round(s.confidence * 100) + "%", "confidence"),
    ].join("");
    const w = $("#warnings");
    w.classList.toggle("hidden", !s.warnings.length);
    w.innerHTML = s.warnings.map((x) => "⚠ " + esc(x)).join("<br>");

    $("#plan").innerHTML = svg;
    renderCompass(model, room);
    renderGrid(room);
    renderObjects(room, model.north_deg);
    const meshUrl = job.links && job.links.mesh;
    $("#meshTab").classList.toggle("hidden", !meshUrl);
    if (meshUrl) $("#mv").setAttribute("src", meshUrl); else $("#mv").removeAttribute("src");
    $("#json").textContent = JSON.stringify(model, null, 2);
    $("#dlJson").href = URL.createObjectURL(new Blob([JSON.stringify(model, null, 2)], { type: "application/json" }));
    $("#dlJson").download = `${job.id}.json`;
    $("#dlSvg").href = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    $("#dlSvg").download = `${job.id}.svg`;
    const maxMs = Math.max(...s.stages.map((x) => x.duration_ms), 1);
    $("#stageTable").innerHTML = "<tr><th>Stage</th><th>Status</th><th>Time</th><th style='width:40%'></th></tr>" + s.stages.map((x) =>
      `<tr><td>${esc(STAGE_LABEL[x.name] || x.name)}</td><td>${esc(x.status)}</td><td class="num">${fmtMs(x.duration_ms)}</td>` +
      `<td><div class="bar" style="width:${Math.max(1, (x.duration_ms / maxMs) * 100)}%"></div></td></tr>`).join("");
    showRTab("plan");
  }
  const fmtMs = (ms) => (ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms");

  function bearingOf(dx, dy, north) { return ((north + (Math.atan2(dx, dy) * 180) / Math.PI) % 360 + 360) % 360; }

  function renderCompass(model, room) {
    const N = model.north_deg;
    const el = $("#compass");
    if (N == null) { el.innerHTML = "<p class='hint'>No north heading was supplied, so directions are unknown.</p>"; return; }
    const S = 440, c = S / 2, R = 180;
    const pt = (b, r) => [c + r * Math.sin((b * Math.PI) / 180), c - r * Math.cos((b * Math.PI) / 180)];
    const parts = [`<svg viewBox="0 0 ${S} ${S}" role="img" aria-label="Vastu compass">`];
    for (let i = 0; i < 16; i++) {
      const a0 = i * 22.5 - 11.25, a1 = a0 + 22.5;
      const [x0, y0] = pt(a0, R), [x1, y1] = pt(a1, R);
      parts.push(`<path d="M${c},${c} L${x0},${y0} A${R},${R} 0 0 1 ${x1},${y1} Z" fill="${i % 2 ? "var(--soft)" : "var(--card)"}" stroke="var(--line)"/>`);
      const [lx, ly] = pt(i * 22.5, R + 22);
      const major = DIRS16[i].length === 1;
      parts.push(`<text x="${lx}" y="${ly + 4}" text-anchor="middle" font-size="${major ? 14 : 10.5}" font-weight="${major ? 700 : 500}" fill="${DIRS16[i] === "N" ? "var(--accent)" : "var(--muted)"}">${DIRS16[i]}</text>`);
    }
    parts.push(`<circle cx="${c}" cy="${c}" r="${R * 0.33}" fill="none" stroke="var(--accent)" stroke-dasharray="4 4" opacity=".6"/>`);
    parts.push(`<text x="${c}" y="${c + 4}" text-anchor="middle" font-size="10.5" fill="var(--muted)">Brahmasthan</text>`);
    const [cx, cy] = room.centroid;
    const dist = room.objects.map((o) => Math.hypot(o.position[0] - cx, o.position[1] - cy));
    const maxD = Math.max(...room.floor_polygon.map(([x, y]) => Math.hypot(x - cx, y - cy)), 0.1);
    room.openings.forEach((op) => {
      const b = bearingOf(op.center[0] - cx, op.center[1] - cy, N);
      const [x, y] = pt(b, R - 6);
      parts.push(`<rect x="${x - 7}" y="${y - 7}" width="14" height="14" rx="3" fill="${op.type === "door" ? "#d35400" : "#3498db"}"><title>${op.type} · ${dirOf(b)}</title></rect>`);
    });
    room.objects.forEach((o, i) => {
      const b = bearingOf(o.position[0] - cx, o.position[1] - cy, N);
      const r = 20 + (dist[i] / maxD) * (R - 40);
      const [x, y] = pt(b, r);
      parts.push(`<circle cx="${x}" cy="${y}" r="7" fill="${ELEMENT[o.element] || "#666"}" stroke="#fff" stroke-width="1.5"><title>${esc(o.label)} · ${o.direction_16}</title></circle>`);
      parts.push(`<text x="${x + 10}" y="${y + 4}" font-size="11.5" fill="var(--ink)" style="paint-order:stroke;stroke:var(--card);stroke-width:3px">${esc(o.label)}</text>`);
    });
    parts.push("</svg>");
    const legend = Object.keys(ELEMENT).map((k) => `<span class="chip"><i style="background:${ELEMENT[k]}"></i>${k}</span>`).join("") +
      `<span class="chip"><i style="background:#d35400;border-radius:2px"></i>door</span><span class="chip"><i style="background:#3498db;border-radius:2px"></i>window</span>`;
    el.innerHTML = parts.join("") + `<div class="legend">${legend}</div>`;
  }
  const dirOf = (b) => DIRS16[Math.floor(((b + 11.25) % 360) / 22.5)];

  function renderGrid(room) {
    const order = [["NW", "N", "NE"], ["W", "BRAHMASTHAN", "E"], ["SW", "S", "SE"]];
    const by = {};
    room.objects.forEach((o) => { (by[o.grid_zone] = by[o.grid_zone] || []).push(o); });
    $("#grid9").innerHTML = order.flat().map((z) => `<div class="cell ${z === "BRAHMASTHAN" ? "center" : ""}"><div class="z">${z === "BRAHMASTHAN" ? "CENTRE" : z}</div>` +
      (by[z] || []).map((o) => `<span class="chip"><i style="background:${ELEMENT[o.element] || "#666"}"></i>${esc(o.label)}</span>`).join("") + "</div>").join("");
  }

  function renderObjects(room, north) {
    const rows = room.objects.map((o) => `<tr><td>${esc(o.label)}</td><td>${esc(o.category)}</td><td><span class="chip"><i style="background:${ELEMENT[o.element] || "#999"}"></i>${esc(o.element || "–")}</span></td>` +
      `<td>${esc(o.direction_16 || "–")}</td><td>${esc(o.grid_zone === "BRAHMASTHAN" ? "Centre" : o.grid_zone || "–")}</td>` +
      `<td class="num">${o.position.slice(0, 2).map((v) => v.toFixed(2)).join(", ")}</td><td class="num">${Math.round(o.confidence * 100)}%</td></tr>`);
    const [cx, cy] = room.centroid;
    const ops = room.openings.map((op) => `<tr><td>${op.type}</td><td>opening</td><td>–</td>` +
      `<td>${north == null ? "–" : dirOf(bearingOf(op.center[0] - cx, op.center[1] - cy, north))}</td><td>–</td>` +
      `<td class="num">${op.width_m.toFixed(2)} × ${op.height_m.toFixed(2)} m</td><td></td></tr>`);
    $("#objTable").innerHTML = "<tr><th>Item</th><th>Category</th><th>Element</th><th>Direction</th><th>Grid zone</th><th>Position / size (m)</th><th>Conf.</th></tr>" +
      (rows.concat(ops).join("") || "<tr><td colspan='7' class='muted'>No objects detected.</td></tr>");
  }

  // --------------------------------------------------------------- history
  const TIER_NAME = { panorama: "360° photo", lidar: "LiDAR", arcore: "ARCore" };
  async function loadHistory() {
    try {
      const jobs = await api("/v1/scans?limit=8");
      const ul = $("#history");
      if (!jobs.length) { ul.innerHTML = '<li class="muted">Nothing yet.</li>'; return; }
      ul.innerHTML = jobs.map((j) => `<li data-id="${esc(j.id)}"><span class="pill ${esc(j.status)}">${esc(j.status)}</span>` +
        `<span>${esc((j.meta && j.meta.room_name) || TIER_NAME[j.tier])}</span><span class="meta">${esc(TIER_NAME[j.tier] || j.tier)} · ${ago(j.created_at)}</span></li>`).join("");
      $$("#history li[data-id]").forEach((li) => li.addEventListener("click", () => {
        const j = jobs.find((x) => x.id === li.dataset.id);
        showProgress(null, j);
        track(j.id, null);
      }));
    } catch { /* API down: status pill already says so */ }
  }
  function ago(t) {
    const s = Date.now() / 1000 - t;
    if (s < 60) return "just now";
    if (s < 3600) return Math.round(s / 60) + " min ago";
    if (s < 86400) return Math.round(s / 3600) + " h ago";
    return Math.round(s / 86400) + " d ago";
  }

  $("#apiDocs").href = API + "/docs";
  health();
  loadHistory();
})();
