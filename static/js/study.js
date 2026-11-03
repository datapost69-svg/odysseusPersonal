// Local Study workspace — lazy, native Odysseus panel.
const $ = (id) => document.getElementById(id);
let modules = [];
let activeModule = "";
let mediaStream = null;
let recorder = null;
let chunks = [];
let recordingStarted = 0;
let timer = null;

function toast(message) {
  const el = $("study-status");
  if (el) el.textContent = message;
}
async function api(url, options = {}) {
  const response = await fetch(url, { credentials: "same-origin", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || `Request failed (${response.status})`);
  return data;
}
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function renderModules() {
  const select = $("study-module");
  if (!select) return;
  select.innerHTML = '<option value="">Choose a module…</option>' +
    modules.map(m => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("");
  select.value = activeModule;
  $("study-empty")?.classList.toggle("hidden", modules.length > 0);
  $("study-workspace")?.classList.toggle("hidden", modules.length === 0);
}
async function loadModules() {
  const data = await api("/api/study/modules");
  modules = data.modules || [];
  if (!modules.some(m => m.id === activeModule)) activeModule = modules[0]?.id || "";
  renderModules();
  if (activeModule) { await loadLectures(); await loadSources(); }
}
async function loadLectures() {
  if (!activeModule) return;
  const data = await api(`/api/study/modules/${encodeURIComponent(activeModule)}/lectures`);
  const list = $("study-lecture-list");
  if (!list) return;
  list.innerHTML = data.lectures.length ? data.lectures.map(l =>
    `<button class="study-lecture-row" data-lecture-id="${esc(l.id)}"><span>${esc(l.title)}</span><small>${esc(l.language || "Language unknown")} · ${Math.round((l.duration || 0) / 60)} min</small></button>`
  ).join("") : '<div class="study-muted">No lecture recordings yet.</div>';
  list.querySelectorAll("[data-lecture-id]").forEach(btn => btn.addEventListener("click", () => openLecture(btn.dataset.lectureId)));
}
async function loadSources() {
  if (!activeModule) return;
  const data = await api(`/api/study/modules/${encodeURIComponent(activeModule)}/sources`);
  const list = $("study-source-list");
  if (!list) return;
  list.innerHTML = data.sources.length ? data.sources.map(s =>
    `<div class="study-lecture-row"><span>${esc(s.title)}</span><small>${esc(s.format.toUpperCase())} · ${s.pages} sections</small></div>`
  ).join("") : '<div class="study-muted">No course materials added yet.</div>';
}
async function importSource(file) {
  if (!file) return;
  if (!activeModule) { toast("Choose a module first."); return; }
  const form = new FormData();
  form.append("module_id", activeModule);
  form.append("file", file, file.name);
  toast("Extracting and indexing locally…");
  try {
    const data = await api("/api/study/sources", {method:"POST", body:form});
    await loadSources();
    toast(`Added ${data.source.title} to this module.`);
  } catch (error) { toast(error.message); }
}
async function searchLibrary() {
  const query = $("study-search-query").value.trim();
  if (!query) return toast("Enter something to search for.");
  const list = $("study-search-results");
  list.innerHTML = '<div class="study-muted">Searching your local library…</div>';
  try {
    const data = await api(`/api/study/search?q=${encodeURIComponent(query)}`);
    list.innerHTML = data.results.length ? data.results.map(r =>
      `<article class="study-search-result"><div><strong>${esc(r.source)}</strong><small>${esc(r.module)} · ${esc(r.location)}</small></div><p>${esc(r.snippet)}</p></article>`
    ).join("") : '<div class="study-muted">No matching passages found.</div>';
  } catch (error) { list.innerHTML = `<div class="study-muted">${esc(error.message)}</div>`; }
}

async function openLecture(id) {
  const data = await api(`/api/study/modules/${encodeURIComponent(activeModule)}/lectures/${encodeURIComponent(id)}`);
  $("study-transcript-title").textContent = data.title;
  $("study-transcript").innerHTML = (data.segments || []).map(s =>
    `<div class="study-segment"><time>${formatTime(s.start)}</time><span>${esc(s.text)}</span></div>`
  ).join("");
  $("study-transcript-panel").classList.remove("hidden");
}
function formatTime(seconds) {
  const n = Math.max(0, Math.floor(seconds || 0));
  return `${String(Math.floor(n / 60)).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`;
}
async function createModule() {
  const name = $("study-new-module").value.trim();
  if (!name) return toast("Enter a module name first.");
  const code = $("study-module-code").value.trim();
  const data = await api("/api/study/modules", {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({name, code})
  });
  modules.push(data.module);
  activeModule = data.module.id;
  $("study-new-module").value = "";
  $("study-module-code").value = "";
  renderModules();
  await loadLectures();
  toast("Module created locally.");
}
function stopCaptureTracks() {
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  mediaStream = null;
}
async function startCapture() {
  if (!activeModule) return toast("Choose a module first.");
  if (!navigator.mediaDevices?.getDisplayMedia) {
    return toast("Live system-audio capture is unavailable here. Use Import audio instead.");
  }
  try {
    // Chromium's desktop picker exposes the Windows system-audio checkbox
    // when sharing an entire screen. The video track is discarded immediately.
    mediaStream = await navigator.mediaDevices.getDisplayMedia({video: true, audio: true});
    const audioTracks = mediaStream.getAudioTracks();
    if (!audioTracks.length) {
      stopCaptureTracks();
      return toast("No system audio was shared. Select a screen and enable 'Share system audio', then try again.");
    }
    mediaStream.getVideoTracks().forEach(t => { t.stop(); mediaStream.removeTrack(t); });
    const mime = ["audio/webm;codecs=opus", "audio/webm"].find(t => MediaRecorder.isTypeSupported(t));
    if (!mime) throw new Error("This browser cannot record WebM audio.");
    chunks = [];
    recorder = new MediaRecorder(mediaStream, {mimeType: mime});
    recorder.ondataavailable = e => { if (e.data?.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      stopCaptureTracks();
      if (!chunks.length) return toast("No audio was captured.");
      const blob = new Blob(chunks, {type: mime});
      await uploadAudio(blob, `live-lecture-${new Date().toISOString().replace(/[:.]/g,"-")}.webm`);
    };
    recorder.start(1000);
    recordingStarted = Date.now();
    $("study-recording").classList.remove("hidden");
    $("study-start").classList.add("hidden");
    $("study-stop").classList.remove("hidden");
    timer = setInterval(() => {
      $("study-recording-time").textContent = formatTime((Date.now() - recordingStarted) / 1000);
    }, 1000);
    mediaStream.getVideoTracks().forEach(t => t.addEventListener("ended", () => stopCapture()));
    toast("Listening to shared system audio. Stop when the lecture finishes.");
  } catch (error) {
    stopCaptureTracks();
    toast(error.message || "Could not start audio capture.");
  }
}
function stopCapture() {
  if (timer) clearInterval(timer);
  timer = null;
  if (recorder && recorder.state !== "inactive") recorder.stop();
  else stopCaptureTracks();
  $("study-recording").classList.add("hidden");
  $("study-stop").classList.add("hidden");
  $("study-start").classList.remove("hidden");
}
async function uploadAudio(blob, filename) {
  const form = new FormData();
  form.append("module_id", activeModule);
  form.append("file", blob, filename);
  toast("Transcribing locally… this can take a while for long lectures.");
  try {
    const data = await api("/api/study/audio", {method:"POST", body:form});
    await loadLectures();
    $("study-transcript-title").textContent = data.lecture.title;
    $("study-transcript").innerHTML = (data.segments || []).map(s =>
      `<div class="study-segment"><time>${formatTime(s.start)}</time><span>${esc(s.text)}</span></div>`
    ).join("");
    $("study-transcript-panel").classList.remove("hidden");
    toast("Transcript saved on this device.");
  } catch (error) {
    toast(error.message);
  }
}
function importAudio(file) {
  if (!file) return;
  if (!activeModule) { toast("Choose a module first."); return; }
  uploadAudio(file, file.name);
}
function openPanel() {
  const panel = $("study-panel");
  panel?.classList.remove("hidden");
  panel?.setAttribute("aria-hidden", "false");
  loadModules().catch(e => toast(e.message));
}
function closePanel() {
  stopCapture();
  $("study-panel")?.classList.add("hidden");
  $("study-panel")?.setAttribute("aria-hidden", "true");
}
function init() {
  $("rail-study")?.addEventListener("click", openPanel);
  $("study-close")?.addEventListener("click", closePanel);
  $("study-panel")?.addEventListener("click", e => { if (e.target.id === "study-panel") closePanel(); });
  $("study-create-module")?.addEventListener("click", () => createModule().catch(e => toast(e.message)));
  $("study-module")?.addEventListener("change", async e => { activeModule = e.target.value; await Promise.all([loadLectures(), loadSources()]).catch(err => toast(err.message)); });
  $("study-start")?.addEventListener("click", startCapture);
  $("study-stop")?.addEventListener("click", stopCapture);
  $("study-import")?.addEventListener("change", e => importAudio(e.target.files?.[0]));
  $("study-refresh")?.addEventListener("click", () => Promise.all([loadLectures(), loadSources()]).catch(e => toast(e.message)));
  $("study-source-import")?.addEventListener("change", e => importSource(e.target.files?.[0]));
  $("study-search-button")?.addEventListener("click", () => searchLibrary());
  $("study-search-query")?.addEventListener("keydown", e => { if (e.key === "Enter") searchLibrary(); });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once:true});
else init();
