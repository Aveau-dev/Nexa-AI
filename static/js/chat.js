window.Chat = (function () {
  let currentChatId = null;
  let currentModelKey = (document.getElementById("model-info")?.textContent || "gpt-3.5-turbo").trim();
  let isLoading = false;

  function onViewLoaded() {
    // Nothing heavy; view HTML must include #messages-container and optional #welcome-section
    const input = document.getElementById("chat-input");
    if (input) input.focus();
  }

  function setModel(modelKey /*, modelName */) {
    currentModelKey = modelKey;
  }

  function setWelcomeVisible(visible) {
    const welcome = document.getElementById("welcome-section");
    if (welcome) welcome.style.display = visible ? "block" : "none";
  }

  function formatMarkdown(text) {
    if (typeof marked === "undefined") return UI.escapeHtml(text);

    marked.setOptions({
      breaks: true,
      gfm: true,
      highlight: function (code, lang) {
        if (typeof hljs === "undefined") return code;
        if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
        return hljs.highlightAuto(code).value;
      },
    });

    return marked.parse(text || "");
  }

  function addMessage(text, role, modelName) {
    const container = document.getElementById("messages-container");
    if (!container) return;

    const row = document.createElement("div");
    row.className = `message-row ${role}-row`;

    if (role === "user") {
      row.innerHTML = `
        <div class="message user-message">
          <div class="message-content">
            <div class="message-text">${UI.escapeHtml(text)}</div>
          </div>
        </div>`;
    } else if (role === "assistant") {
      row.innerHTML = `
        <div class="message assistant-message">
          <div class="avatar">N</div>
          <div class="message-content">
            ${modelName ? `<div class="model-badge">${UI.escapeHtml(modelName)}</div>` : ""}
            <div class="message-text markdown-content">${formatMarkdown(text)}</div>
          </div>
        </div>`;
    } else {
      row.innerHTML = `
        <div class="message error-message">
          <div class="message-content">${UI.escapeHtml(text)}</div>
        </div>`;
    }

    container.appendChild(row);
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  async function newChat() {
    if (isLoading) return;
    isLoading = true;
    try {
      const res = await fetch("/chat/new", { method: "POST" });
      const data = await res.json();
      currentChatId = data.chatid ?? data.chat_id ?? null;

      await Router.go("chat");

      const container = document.getElementById("messages-container");
      if (container) container.innerHTML = "";
      setWelcomeVisible(true);
    } catch (e) {
      alert("Failed to create new chat.");
    } finally {
      isLoading = false;
    }
  }

  async function loadChat(chatId) {
    if (isLoading) return;
    isLoading = true;

    try {
      await Router.go("chat");

      const res = await fetch(`/chat/${chatId}/messages`);
      const data = await res.json();
      currentChatId = chatId;

      const container = document.getElementById("messages-container");
      if (container) container.innerHTML = "";

      setWelcomeVisible(false);
      (data.messages || []).forEach((m) => addMessage(m.content, m.role, m.model));

      document.querySelectorAll(".chat-item").forEach((el) => el.classList.remove("active"));
      const active = document.querySelector(`[data-chat-id="${chatId}"]`);
      if (active) active.classList.add("active");
    } catch (e) {
      alert("Failed to load chat.");
    } finally {
      isLoading = false;
    }
  }

  async function sendMessage() {
    const input = document.getElementById("chat-input");
    if (!input) return;

    const message = input.value.trim();
    if (!message || isLoading) return;

    isLoading = true;
    setWelcomeVisible(false);

    addMessage(message, "user");
    input.value = "";
    input.style.height = "auto";

    try {
      const body = {
        message,
        model: currentModelKey,
        chatid: currentChatId,
      };

      // Attach file path if uploaded via Files module
      if (window.Files && Files.getSelectedUploadPath) {
        const p = Files.getSelectedUploadPath();
        if (p) body.uploadedfile = p;
      }

      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (data.error) {
        addMessage(data.error, "error");
        return;
      }

      addMessage(data.response, "assistant", data.model);
      currentChatId = data.chatid ?? data.chat_id ?? currentChatId;

      // clear selected file after successful send
      if (window.Files && Files.clearSelectedUpload) Files.clearSelectedUpload();
    } catch (e) {
      addMessage("Error connecting to AI. Please try again.", "error");
    } finally {
      isLoading = false;
    }
  }

  function useSuggestion(text) {
    Router.go("chat").then(() => {
      const input = document.getElementById("chat-input");
      if (!input) return;
      input.value = text;
      sendMessage();
    });
  }

  // optional: keep old inline onclick working
  window.newChat = newChat;
  window.loadChat = loadChat;
  window.sendMessage = sendMessage;
  window.useSuggestion = useSuggestion;

  return { onViewLoaded, newChat, loadChat, sendMessage, useSuggestion, setModel };
})();
