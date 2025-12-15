// static/js/modes.js
(function () {
  const STORAGE_KEY = "chat_modes";
  const state = { deepthink: false, web: false };

  function load() {
    try {
      Object.assign(state, JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"));
    } catch {}
  }

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }

  function render() {
    document.getElementById("btn-deepthink")?.classList.toggle("active", !!state.deepthink);
    document.getElementById("btn-web")?.classList.toggle("active", !!state.web);
  }

  function init() {
    load();
    render();

    document.getElementById("btn-deepthink")?.addEventListener("click", () => {
      state.deepthink = !state.deepthink;
      save();
      render();
    });

    document.getElementById("btn-web")?.addEventListener("click", () => {
      state.web = !state.web;
      save();
      render();
    });
  }

  function getModes() {
    return { ...state };
  }

  window.NexaModes = { init, getModes };
})();
