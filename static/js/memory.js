window.Memory = (function () {
  let enabled = false;

  function onViewLoaded() {
    const cb = document.getElementById("memory-enabled");
    if (cb) cb.checked = enabled;
    updateStatus();
  }

  function updateStatus(msg) {
    const el = document.getElementById("memory-status");
    if (!el) return;
    el.textContent = msg || `Memory is ${enabled ? "enabled" : "disabled"}.`;
  }

  function setEnabled(v) {
    enabled = !!v;
    updateStatus();
  }

  function toggle() {
    enabled = !enabled;
    const cb = document.getElementById("memory-enabled");
    if (cb) cb.checked = enabled;
    updateStatus();
  }

  function save() {
    const text = document.getElementById("memory-text")?.value || "";
    updateStatus("Saved (client-side placeholder).");
    const items = document.getElementById("memory-items");
    if (items) items.textContent = text ? `Saved memory:\n${text}` : "No memory saved.";
  }

  function clear() {
    const ta = document.getElementById("memory-text");
    if (ta) ta.value = "";
    const items = document.getElementById("memory-items");
    if (items) items.textContent = "No memory saved.";
    updateStatus("Cleared (client-side placeholder).");
  }

  function refresh() {
    updateStatus();
  }

  return { onViewLoaded, toggle, setEnabled, save, clear, refresh };
})();
