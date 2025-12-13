/* global window, document, fetch */
(function () {
  const Router = {
    async load(viewName) {
      const container = document.getElementById('view-container');
      if (!container) return;

      container.innerHTML = `<div class="loading">Loading…</div>`;

      try {
        const res = await fetch(`/load-view/${encodeURIComponent(viewName)}`, {
          headers: { "X-Requested-With": "fetch" }
        });

        if (!res.ok) {
          container.innerHTML = `<div class="message error-message"><div class="message-content">
            View load failed (${res.status}). Check /load-view/${viewName}.
          </div></div>`;
          return;
        }

        const html = await res.text();
        container.innerHTML = html;

        // hook per-view
        if (window.Chat && viewName === 'chat' && typeof window.Chat.onViewMounted === 'function') {
          window.Chat.onViewMounted();
        }

        container.focus();
      } catch (e) {
        console.error("Router.load error", e);
        container.innerHTML = `<div class="message error-message"><div class="message-content">
          Network error while loading view.
        </div></div>`;
      }
    }
  };

  window.Router = Router;
})();
