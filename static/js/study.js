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
  form.append("material_type", $("study-material-type")?.value || "other");
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
    const data = await api(`/api/study/semantic-search?q=${encodeURIComponent(query)}`);
    list.innerHTML = data.results.length ? data.results.map(r =>
      `<article class="study-search-result"><div><strong>${esc(r.source)}</strong><small>${esc(r.module)} · ${esc(r.location)} · relevance ${Math.round(r.score * 100)}%</small></div><p>${esc(r.snippet)}</p></article>`
    ).join("") : '<div class="study-muted">No related passages found. Add more material or try a different description.</div>';
  } catch (error) { list.innerHTML = `<div class="study-muted">${esc(error.message)}</div>`; }
}

async function loadStudyModels() {
  const select = $("study-model-select");
  if (!select) return;
  select.innerHTML = '<option value="">Checking local Ollama…</option>';
  try {
    const data = await api("/api/study/models");
    if (!data.available || !data.models.length) {
      select.innerHTML = '<option value="">No local Ollama models found</option>';
      toast("Start Ollama and ensure you have a chat model installed to ask questions.");
      return;
    }
    const preferred = data.models.find(name => /qwen|llama|mistral/i.test(name)) || data.models[0];
    select.innerHTML = data.models.map(name =>
      `<option value="${esc(name)}">${esc(name)}</option>`
    ).join("");
    select.value = preferred;
  } catch (error) {
    select.innerHTML = '<option value="">Local Ollama unavailable</option>';
    toast(error.message);
  }
}

async function loadObsidianVault() {
  try {
    const data = await api("/api/study/vault");
    $("study-vault-path").value = data.vault_path || "";
    $("study-vault-status").textContent = data.configured
      ? "Obsidian vault connected on this device."
      : "Choose the folder of your existing Obsidian vault.";
  } catch (error) { $("study-vault-status").textContent = error.message; }
}
async function saveObsidianVault() {
  const vault_path = $("study-vault-path").value.trim();
  if (!vault_path) return toast("Enter your Obsidian vault folder path.");
  const button = $("study-vault-save");
  button.disabled = true;
  try {
    const data = await api("/api/study/vault", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({vault_path})
    });
    $("study-vault-path").value = data.vault_path;
    $("study-vault-status").textContent = "Obsidian vault connected on this device.";
    toast("Obsidian vault location saved locally.");
  } catch (error) { toast(error.message); }
  finally { button.disabled = false; }
}
async function generateStudyNotes() {
  if (!activeModule) return toast("Choose a module first.");
  const model = $("study-model-select").value;
  if (!model) return toast("Choose an installed local Ollama model.");
  const button = $("study-generate-notes");
  const status = $("study-note-status");
  const preview = $("study-note-preview");
  button.disabled = true;
  button.textContent = "Generating…";
  status.textContent = "Retrieving course material and generating notes locally. This may take a few minutes.";
  preview.classList.add("hidden");
  try {
    const data = await api("/api/study/notes/generate", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        module_id:activeModule,
        note_type:$("study-note-type").value,
        brief:$("study-note-brief").value.trim(),
        title:$("study-note-title").value.trim(),
        model
      })
    });
    preview.textContent = data.note;
    preview.classList.remove("hidden");
    status.textContent = "Saved " + data.sources + " cited source passages to " + data.path;
    toast("Notes saved to your local Obsidian vault.");
  } catch (error) {
    status.textContent = error.message;
    toast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Generate and save to Obsidian";
  }
}

async function askLibrary() {
  const question = $("study-question").value.trim();
  const model = $("study-model-select").value;
  if (!question) return toast("Enter a question first.");
  if (!model) return toast("Choose an installed local Ollama model.");
  const button = $("study-ask-button");
  const answer = $("study-answer");
  const sources = $("study-answer-sources");
  button.disabled = true;
  button.textContent = "Thinking…";
  answer.classList.remove("hidden");
  answer.textContent = "Searching your local library and preparing a grounded answer…";
  sources.innerHTML = "";
  try {
    const data = await api("/api/study/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({question, module_id: $("study-ask-all-modules").checked ? null : (activeModule || null), model})
    });
    answer.textContent = data.answer;
    answer.classList.toggle("study-answer-ungrounded", !data.grounded);
    if (data.sources?.length) {
      sources.innerHTML = '<div class="study-label">Sources used</div>' + data.sources.map(source =>
        `<article class="study-source-citation"><div><strong>[${esc(source.marker)}] ${esc(source.source)}</strong><small>${esc(source.module)} · ${esc(source.location)}</small></div><p>${esc(source.snippet)}</p></article>`
      ).join("");
    } else {
      sources.innerHTML = '<div class="study-muted">No sufficiently relevant source passages were found.</div>';
    }
    toast(data.grounded ? "Answer generated using local sources." : "No sufficiently relevant source evidence found.");
  } catch (error) {
    answer.textContent = error.message;
    answer.classList.add("study-answer-ungrounded");
  } finally {
    button.disabled = false;
    button.textContent = "Ask library";
  }
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
  loadStudyModels();
  loadObsidianVault();
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
  $("study-import")?.addEventListener("change", e => { importAudio(e.target.files?.[0]); e.target.value = ""; });
  $("study-refresh")?.addEventListener("click", () => Promise.all([loadLectures(), loadSources()]).catch(e => toast(e.message)));
  $("study-source-import")?.addEventListener("change", e => { importSource(e.target.files?.[0]); e.target.value = ""; });
  $("study-search-button")?.addEventListener("click", () => searchLibrary());
  $("study-ask-button")?.addEventListener("click", () => askLibrary());
  $("study-vault-save")?.addEventListener("click", saveObsidianVault);
  $("study-generate-notes")?.addEventListener("click", generateStudyNotes);
  $("study-question")?.addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askLibrary(); });
  $("study-search-query")?.addEventListener("keydown", e => { if (e.key === "Enter") searchLibrary(); });
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once:true});
else init();
