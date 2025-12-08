/* static/js/memory.js
   Memory & Custom Instructions
*/

const Memory = (function () {
  async function load() {
    try {
      const resp = await fetch("/api/memory");
      if (!resp.ok) return;
      const data = await resp.json();
      document.getElementById("memory-tone").value = data.tone || "default";
      document.getElementById("memory-notes").value = data.notes || "";
      document.getElementById("memory-instructions").value = data.instructions || "";
    } catch (e) {
      console.error("Memory.load error", e);
    }
  }

  async function save() {
    const payload = {
      tone: document.getElementById("memory-tone").value,
      notes: document.getElementById("memory-notes").value,
      instructions: document.getElementById("memory-instructions").value
    };
    try {
      const resp = await fetch("/api/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (data.error) UI.toast("Save failed");
      else {
        UI.toast("Memory saved");
        document.getElementById("memory-status").textContent = "Saved";
        setTimeout(() => document.getElementById("memory-status").textContent = "", 2000);
      }
    } catch (e) {
      console.error("Memory.save error", e);
      UI.toast("Save error");
    }
  }

  return {
    load,
    save
  };
})();

window.Memory = Memory;
