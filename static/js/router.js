/* static/js/router.js */
window.Router = (function () {
  let currentView = null;

  const viewModules = {
    chat: window.Chat,
    files: window.Files,
    memory: window.Memory,
    projects: window.Projects,
    canvas: window.Canvas,
    voice: window.Voice
  };

  function go(viewName) {
    if (currentView === viewName) return;
    currentView = viewName;

    // Dispatch custom event
    document.dispatchEvent(new CustomEvent("nexa:viewchange", { 
      detail: { view: viewName } 
    }));

    // Call onViewLoaded if module exists
    const mod = viewModules[viewName];
    if (mod && typeof mod.onViewLoaded === "function") {
      setTimeout(() => mod.onViewLoaded(), 50);
    }
  }

  function boot() {
    console.log("Router booted");
    // Default to chat view
    if (!currentView) go("chat");
  }

  return { go, boot };
})();
