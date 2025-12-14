// Simple view loader: fetches /view/<name> and injects HTML into #view-container
window.Router = (function () {
  const allowedViews = new Set(["chat", "files", "memory", "projects", "canvas", "voice", "settings"]);

  async function go(viewName) {
    if (!allowedViews.has(viewName)) viewName = "chat";

    const container = document.getElementById("view-container");
    if (!container) return;

    // Show chat input only for chat view
    const inputWrap = document.getElementById("chat-input-wrapper");
    if (inputWrap) inputWrap.style.display = viewName === "chat" ? "block" : "none";

    container.innerHTML = `
      <div class="welcome-section">
        <h1 class="main-heading">Loading...</h1>
      </div>
    `;

    const res = await fetch(`/view/${viewName}`, { headers: { "X-Requested-With": "fetch" } });
    if (!res.ok) {
      container.innerHTML = `
        <div class="welcome-section">
          <h1 class="main-heading">View not found</h1>
          <p class="footer-text">${UI.escapeHtml(viewName)}</p>
        </div>
      `;
      return;
    }

    const html = await res.text();
    container.innerHTML = html;

    // view hooks
    if (viewName === "chat") Chat.onViewLoaded();
    if (viewName === "files") Files.onViewLoaded();
    if (viewName === "memory") Memory.onViewLoaded();
    if (viewName === "projects") Projects.onViewLoaded();
    if (viewName === "canvas") Canvas.onViewLoaded();
    if (viewName === "voice") Voice.onViewLoaded();
    if (viewName === "settings") Settings.onViewLoaded();

    // close mobile sidebar after navigation
    const sidebar = document.getElementById("sidebar");
    if (sidebar && window.innerWidth <= 1024) sidebar.classList.remove("show");

    // simple hash routing
    try { location.hash = `#${viewName}`; } catch (_) {}
  }

  function boot() {
    const hash = (location.hash || "").replace("#", "").trim();
    go(hash || "chat");
  }

  window.addEventListener("hashchange", boot);

  return { go, boot };
})();
