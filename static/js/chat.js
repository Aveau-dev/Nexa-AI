window.Chat = (function () {
  let currentChatId = null;
  let currentModelKey = (document.getElementById("model-info")?.textContent || "gpt-3.5-turbo").trim();
  let isLoading = false;

  function onViewLoaded() {
    // Ensure textarea listeners exist (main.js also does this globally)
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

      // ensure we are on chat view
      await Router.go("chat");

      // clear messages
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

      (data.messages || []).forEach((m) => {
        addMessage(m.content, m.role, m.model);
      });

      // sidebar highlight
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
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          model: currentModelKey,
          chatid: currentChatId,
          uploadedfile: Files.getSelectedUploadPath ? Files.getSelectedUploadPath() : null,
        }),
      });

      const data = await res.json();

      if (data.error) {
        addMessage(data.error, "error");
        return;
      }

      addMessage(data.response, "assistant", data.model);

      currentChatId = data.chatid ?? data.chat_id ?? currentChatId;

      // update DeepSeek usage if provided
      if (data.deepseekremaining !== undefined && data.deepseekremaining !== null) {
        const usageValue = document.querySelector(".usage-value");
        if (usageValue) usageValue.textContent = `${50 - data.deepseekremaining}/50`;
      }
    } catch (e) {
      addMessage("Error connecting to AI. Please try again.", "error");
    } finally {
      // once message is sent, clear selected upload
      if (Files.clearSelectedUpload) Files.clearSelectedUpload();
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

  // Backward compatibility for inline onclick in older templates
  window.newChat = newChat;
  window.loadChat = loadChat;
  window.sendMessage = sendMessage;
  window.useSuggestion = useSuggestion;

  return {
    onViewLoaded,
    newChat,
    loadChat,
    sendMessage,
    useSuggestion,
    setModel,
    setCurrentChatId: (id) => (currentChatId = id),
  };
})();
