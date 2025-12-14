document.addEventListener("DOMContentLoaded", () => {
  // Boot router
  if (window.Router && Router.boot) Router.boot();

  // Auto-resize chat textarea (exists in dashboard shell)
  const input = document.getElementById("chat-input");
  if (input) {
    input.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 200) + "px";
    });

    // Enter to send (Shift+Enter newline)
    input.addEventListener("keydown", function (e) {
      // respect Settings enter-to-send if present
      let enterToSend = true;
      try {
        const s = JSON.parse(localStorage.getItem("nexa_settings") || "{}");
        enterToSend = s.enterToSend !== false;
      } catch (_) {}

      if (!enterToSend) return;

      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (window.Chat && Chat.sendMessage) Chat.sendMessage();
      }
    });
  }
});
