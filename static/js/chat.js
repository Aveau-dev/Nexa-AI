window.Chat = (function () {
  let currentChatId = null;
  let currentModelKey = (document.getElementById("model-info")?.textContent || "gpt-3.5-turbo").trim();
  let isLoading = false;

  function onViewLoaded() {
    const input = document.getElementById("chat-input");
    if (input) input.focus();
  }

  function setModel(modelKey) {
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

  function ensureChatListEmptyStateRemoved() {
    const history = document.querySelector(".chat-history");
    if (!history) return;
    const empty = history.querySelector(".empty-state");
    if (empty) empty.remove();
  }

  function addChatToSidebar(chatId, title, makeActive = true) {
    const history = document.querySelector(".chat-history");
    if (!history) return;

    ensureChatListEmptyStateRemoved();

    const item = document.createElement("div");
    item.className = "chat-item";
    item.setAttribute("data-chat-id", chatId);

    item.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"></path>
      </svg>
      <span class="chat-title">${UI.escapeHtml(title || "New Chat")}</span>
      <div class="chat-actions">
        <button class="icon-btn" title="Rename" onclick="event.stopPropagation(); Chat.renameChat(${chatId});">✎</button>
        <button class="icon-btn" title="Delete" onclick="event.stopPropagation(); Chat.deleteChat(${chatId});">🗑</button>
      </div>
    `;

    item.onclick = () => loadChat(chatId);

    // prepend
    history.insertBefore(item, history.firstChild);

    if (makeActive) setActiveChat(chatId);
  }

  function setActiveChat(chatId) {
    document.querySelectorAll(".chat-item").forEach((el) => el.classList.remove("active"));
    const active = document.querySelector(`[data-chat-id="${chatId}"]`);
    if (active) active.classList.add("active");
  }

  async function newChat() {
    if (isLoading) return;
    isLoading = true;
    try {
      const res = await fetch("/chat/new", { method: "POST" });
      const data = await res.json();

      if (!data.success && data.error) throw new Error(data.error);

      currentChatId = data.chatid ?? data.chat_id ?? null;

      await Router.go("chat");

      const container = document.getElementById("messages-container");
      if (container) container.innerHTML = "";
      setWelcomeVisible(true);

      // add to sidebar immediately
      if (currentChatId) addChatToSidebar(currentChatId, data.title || "New Chat", true);
    } catch (e) {
      alert(e.message || "Failed to create new chat.");
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
      if (data.error) throw new Error(data.error);

      currentChatId = chatId;

      const container = document.getElementById("messages-container");
      if (container) container.innerHTML = "";

      setWelcomeVisible(false);
      (data.messages || []).forEach((m) => addMessage(m.content, m.role, m.model));

      setActiveChat(chatId);
    } catch (e) {
      alert(e.message || "Failed to load chat.");
    } finally {
      isLoading = false;
    }
  }

  async function renameChat(chatId) {
    const newTitle = prompt("Enter new chat title:");
    if (!newTitle || !newTitle.trim()) return;

    try {
      const res = await fetch(`/chat/${chatId}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle.trim() }),
      });
      const data = await res.json();
      if (!data.success && data.error) throw new Error(data.error);

      const el = document.querySelector(`[data-chat-id="${chatId}"] .chat-title`);
      if (el) el.textContent = newTitle.trim();
    } catch (e) {
      alert(e.message || "Failed to rename chat.");
    }
  }

  async function deleteChat(chatId) {
    if (!confirm("Delete this chat? This cannot be undone.")) return;

    try {
      const res = await fetch(`/chat/${chatId}/delete`, { method: "DELETE" });
      const data = await res.json();
      if (!data.success && data.error) throw new Error(data.error);

      const item = document.querySelector(`[data-chat-id="${chatId}"]`);
      if (item) item.remove();

      if (currentChatId === chatId) {
        currentChatId = null;
        await Router.go("chat");
        const container = document.getElementById("messages-container");
        if (container) container.innerHTML = "";
        setWelcomeVisible(true);
      }
    } catch (e) {
      alert(e.message || "Failed to delete chat.");
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
      const body = { message, model: currentModelKey, chatid: currentChatId };

      // attach uploaded file path if present
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

      // server may create chat automatically if chatid was null
      currentChatId = data.chatid ?? data.chat_id ?? currentChatId;

      // update sidebar title if server returns one
      if (data.chattitle && currentChatId) {
        const existing = document.querySelector(`[data-chat-id="${currentChatId}"] .chat-title`);
        if (existing) existing.textContent = data.chattitle;
        else addChatToSidebar(currentChatId, data.chattitle, true);
      }

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

  // Make old inline onclick work
  window.newChat = newChat;
  window.loadChat = loadChat;
  window.sendMessage = sendMessage;
  window.useSuggestion = useSuggestion;

  return {
    onViewLoaded,
    setModel,
    newChat,
    loadChat,
    renameChat,
    deleteChat,
    sendMessage,
    useSuggestion,
  };
})();
