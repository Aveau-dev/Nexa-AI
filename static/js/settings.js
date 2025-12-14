window.Settings = (function () {
  const KEY = "nexa_settings";

  function onViewLoaded() {
    const saved = load();
    const compact = !!saved.compact;
    const enterSend = saved.enterToSend !== false;

    const c = document.getElementById("setting-compact");
    const e = document.getElementById("setting-enter-send");
    if (c) c.checked = compact;
    if (e) e.checked = enterSend;

    apply(compact);
    setStatus("Status: loaded.");
  }

  function setStatus(msg) {
    const el = document.getElementById("settings-status");
    if (el) el.textContent = msg;
  }

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || "{}"); }
    catch (_) { return {}; }
  }

  function saveObj(obj) {
    localStorage.setItem(KEY, JSON.stringify(obj));
  }

  function apply(compact) {
    document.body.classList.toggle("ultra-compact", !!compact);
  }

  function setCompact(v) {
    const s = load();
    s.compact = !!v;
    saveObj(s);
    apply(s.compact);
    setStatus("Status: compact updated.");
  }

  function setEnterToSend(v) {
    const s = load();
    s.enterToSend = !!v;
    saveObj(s);
    setStatus("Status: enter-to-send updated.");
  }

  function toggleCompact() {
    const s = load();
    setCompact(!s.compact);
    const cb = document.getElementById("setting-compact");
    if (cb) cb.checked = !!load().compact;
  }

  function save() {
    setStatus("Status: saved.");
  }

  function reset() {
    localStorage.removeItem(KEY);
    apply(false);
    const c = document.getElementById("setting-compact");
    const e = document.getElementById("setting-enter-send");
    if (c) c.checked = false;
    if (e) e.checked = true;
    setStatus("Status: reset.");
  }

  return { onViewLoaded, setCompact, setEnterToSend, toggleCompact, save, reset };
})();
