window.Settings = (function () {
  const KEY = "nexa_settings";
  const DEFAULTS = { compact: false, enterToSend: true };

  function setStatus(msg) {
    const el = document.getElementById("settings-status");
    if (el) el.textContent = msg;
  }

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(KEY) || "{}");
      return {
        compact: !!raw.compact,
        enterToSend: raw.enterToSend !== false // default true
      };
    } catch (_) {
      return { ...DEFAULTS };
    }
  }

  function saveObj(obj) {
    localStorage.setItem(KEY, JSON.stringify(obj));
  }

  function applyUI(s) {
    // compact mode
    document.body.classList.toggle("ultra-compact", !!s.compact);

    // sync checkboxes if they exist
    const c = document.getElementById("setting-compact");
    const e = document.getElementById("setting-enter-send");
    if (c) c.checked = !!s.compact;
    if (e) e.checked = !!s.enterToSend;
  }

  function get() {
    return load();
  }

  function onViewLoaded() {
    const s = load();
    applyUI(s);
    setStatus("Status: loaded.");
  }

  function setCompact(v) {
    const s = load();
    s.compact = !!v;
    saveObj(s);
    applyUI(s);
    setStatus("Status: compact updated.");
  }

  function setEnterToSend(v) {
    const s = load();
    s.enterToSend = !!v;
    saveObj(s);
    applyUI(s);
    setStatus("Status: enter-to-send updated.");
  }

  function toggleCompact() {
    setCompact(!load().compact);
  }

  function reset() {
    localStorage.removeItem(KEY);
    applyUI({ ...DEFAULTS });
    setStatus("Status: reset.");
  }

  return { onViewLoaded, get, setCompact, setEnterToSend, toggleCompact, reset };
})();
