/* static/js/chat.js */
(() => {
  let currentModelKey = (window.NEXAAI && window.NEXAAI.initialModelKey) || "gemini-flash";
  let currentModelName = (window.NEXAAI && window.NEXAAI.initialModelName) || "Gemini 2.5 Flash";

  let currentChatId = null;
  let isLoading = false;

  let deepThinkEnabled = false;
  let webEnabled = false;

  function $(id){ return document.getElementById(id); }

  function bindOnce() {
    // toggles
    const dt = $("btn-deepthink");
    const wb = $("btn-web");
    const sendBtn = $("send-btn");
    const input = $("chat-input");

    if (dt && !dt.dataset.bound) {
      dt.dataset.bound = "1";
      dt.addEventListener("click", () => {
        deepThinkEnabled = !deepThinkEnabled;
        dt.textContent = deepThinkEnabled ? "DeepThink: On" : "DeepThink: Off";
        dt.classList.toggle("active", deepThinkEnabled);
      });
    }

    if (wb && !wb.dataset.bound) {
      wb.dataset.bound = "1";
      wb.addEventListener("click", () => {
        webEnabled = !webEnabled;
        wb.textContent = webEnabled ? "Web: On" : "Web: Off";
        wb.classList.toggle("active", webEnabled);
      });
    }

    if (sendBtn && !sendBtn.dataset.bound) {
      sendBtn.dataset.bound = "1";
      sendBtn.addEventListener("click", () => window.sendMessage());
    }

    if (input && !input.dataset.bound) {
      input.dataset.bound = "1";

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          window.sendMessage();
        }
      });

      input.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 200) + "px";
      });
    }

    // labels
    const top = $("selected-model-name");
    const foot = $("model-info");
    if (top) top.textContent = currentModelName;
    if (foot) foot.textContent = currentModelName;

    // ensure input is focusable
    if (input) input.removeAttribute("disabled");
  }

  // Router replaces #view-container; we must re-bind after view loads.
  // If your router dispatches an event, hook it. Otherwise, poll lightly.
  setInterval(bindOnce, 300);

  // ---- global UI funcs used by HTML ----
  window.toggleSidebar = function () {
    const sidebar = $("sidebar");
    const main = $("main-content");
    if (!sidebar || !main) return;

    if (window.innerWidth < 1024) sidebar.classList.toggle("show");
    else { sidebar.classList.toggle("collapsed"); main.classList.toggle("sidebar-collapsed"); }
  };

  window.toggleUserMenu = function () {
    const menu = $("user-menu");
    if (!menu) return;
    menu.style.display = (menu.style.display === "block") ? "none" : "block";
  };

  window.toggleModelSelector = function () {
    const selector = $("model-selector");
    if (!selector) return;
    selector.style.display = (selector.style.display === "block") ? "none" : "block";
  };

  window.selectModel = async function (modelKey, modelName) {
    currentModelKey = modelKey;
    currentModelName = modelName || modelKey;

    const top = $("selected-model-name");
    const foot = $("model-info");
    if (top) top.textContent = currentModelName;
    if (foot) foot.textContent = currentModelName;

    document.querySelectorAll(".model-card").forEach(c => c.classList.remove("active"));
    const active = document.querySelector(`[data-model="${CSS.escape(modelKey)}"]`);
    if (active) active.classList.add("active");

    try {
      await fetch("/set-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelKey })
      });
    } catch (_) {}

    window.toggleModelSelector();
  };

  // ---- helpers ----
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  function formatMessage(text) {
    if (typeof marked === "undefined") return escapeHtml(text).replace(/\n/g, "<br>");
    marked.setOptions({
      highlight(code, lang) {
        if (typeof hljs !== "undefined" && lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return (typeof hljs !== "undefined") ? hljs.highlightAuto(code).value : code;
      },
      breaks: true,
      gfm: true
    });
    return marked.parse(text || "");
  }

  function addMessage(text, role, modelNameForBadge) {
    const container = $("messages-container");
    if (!container) return;

    const row = document.createElement("div");
    row.className = `message-row ${role}-row`;

    if (role === "user") {
      row.innerHTML = `<div class="message user-message"><div class="message-content">${escapeHtml(text)}</div></div>`;
    } else if (role === "assistant") {
      row.innerHTML = `
        <div class="message assistant-message">
          <div class="avatar">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
              <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
            </svg>
          </div>
          <div class="message-content">
            ${modelNameForBadge ? `<div class="model-badge">${escapeHtml(modelNameForBadge)}</div>` : ""}
            <div class="markdown-content"></div>
          </div>
        </div>
      `;
      const contentDiv = row.querySelector(".markdown-content");
      contentDiv.innerHTML = formatMessage(text);
      if (typeof hljs !== "undefined") {
        contentDiv.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
      }
    } else {
      row.innerHTML = `<div class="message error-message"><div class="message-content">${escapeHtml(text)}</div></div>`;
    }

    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }

  function showTypingIndicator() {
    const container = $("messages-container");
    if (!container || $("typing-indicator")) return;

    const typingDiv = document.createElement("div");
    typingDiv.className = "message-row assistant-row typing-indicator";
    typingDiv.id = "typing-indicator";
    typingDiv.innerHTML = `
      <div class="message assistant-message">
        <div class="avatar">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
          </svg>
        </div>
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    `;
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
  }

  function removeTypingIndicator() {
    const el = $("typing-indicator");
    if (el) el.remove();
  }

  function setWelcomeVisible(visible) {
    const w = $("welcome-section");
    if (w) w.style.display = visible ? "block" : "none";
  }

  function setActiveChatInSidebar(chatId) {
    document.querySelectorAll(".chat-item").forEach(el => el.classList.remove("active"));
    const active = document.querySelector(`[data-chat-id="${chatId}"]`);
    if (active) active.classList.add("active");
  }

  function upsertChatInSidebar(chatId, title, prepend) {
    const history = $("chat-history");
    if (!history) return;

    const empty = history.querySelector(".empty-state");
    if (empty) empty.remove();

    let item = history.querySelector(`[data-chat-id="${chatId}"]`);
    if (!item) {
      item = document.createElement("div");
      item.className = "chat-item";
      item.setAttribute("data-chat-id", chatId);
      item.onclick = () => window.loadChat(chatId);

      item.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
          <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"></path>
        </svg>
        <span class="chat-title"></span>
        <div class="chat-actions">
          <button class="icon-btn" title="Rename"></button>
          <button class="icon-btn" title="Delete"></button>
        </div>
      `;

      const buttons = item.querySelectorAll("button.icon-btn");
      buttons[0].onclick = (e) => { e.stopPropagation(); window.renameChat(chatId); };
      buttons[1].onclick = (e) => { e.stopPropagation(); window.deleteChat(chatId); };

      if (prepend && history.firstChild) history.insertBefore(item, history.firstChild);
      else history.appendChild(item);
    }

    const t = item.querySelector(".chat-title");
    if (t) t.textContent = title || "New Chat";
  }

  async function apiJson(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  // ---- chat CRUD ----
  window.newChat = async function () {
    if (isLoading) return;
    isLoading = true;
    try {
      const data = await apiJson("/chat/new", { method: "POST", headers: { "Content-Type": "application/json" } });
      currentChatId = data.chatid;

      const container = $("messages-container");
      if (container) container.innerHTML = "";
      setWelcomeVisible(true);

      upsertChatInSidebar(currentChatId, data.title || "New Chat", true);
      setActiveChatInSidebar(currentChatId);
    } catch (e) {
      addMessage(e.message || "Failed to create new chat.", "error");
    } finally {
      isLoading = false;
    }
  };

  window.loadChat = async function (chatId) {
    if (isLoading) return;
    isLoading = true;
    try {
      const data = await apiJson(`/chat/${chatId}/messages`, { method: "GET" });
      currentChatId = chatId;

      const container = $("messages-container");
      if (container) container.innerHTML = "";
      setWelcomeVisible(false);

      (data.messages || []).forEach(m => addMessage(m.content, m.role, m.model));
      setActiveChatInSidebar(chatId);

      if (window.innerWidth < 1024) {
        const sidebar = $("sidebar");
        if (sidebar) sidebar.classList.remove("show");
      }
    } catch (e) {
      addMessage(e.message || "Failed to load chat.", "error");
    } finally {
      isLoading = false;
    }
  };

  window.renameChat = async function (chatId) {
    const newTitle = prompt("Enter new chat title:");
    if (!newTitle || !newTitle.trim()) return;

    try {
      await apiJson(`/chat/${chatId}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle.trim() })
      });
      upsertChatInSidebar(chatId, newTitle.trim(), false);
    } catch (e) {
      addMessage(e.message || "Failed to rename chat.", "error");
    }
  };

  window.deleteChat = async function (chatId) {
    const ok = confirm("Delete this chat? This cannot be undone.");
    if (!ok) return;

    try {
      await apiJson(`/chat/${chatId}/delete`, { method: "DELETE" });

      const item = document.querySelector(`[data-chat-id="${chatId}"]`);
      if (item) item.remove();

      if (currentChatId === chatId) {
        currentChatId = null;
        const container = $("messages-container");
        if (container) container.innerHTML = "";
        setWelcomeVisible(true);
      }
    } catch (e) {
      addMessage(e.message || "Failed to delete chat.", "error");
    }
  };

  // ---- send ----
  window.sendMessage = async function () {
    if (isLoading) return;

    const input = $("chat-input");
    const message = (input && input.value ? input.value : "").trim();
    if (!message) return;

    isLoading = true;
    setWelcomeVisible(false);
    addMessage(message, "user");

    input.value = "";
    input.style.height = "auto";

    showTypingIndicator();

    try {
      const payload = {
        message,
        model: currentModelKey,
        chatid: currentChatId,
        deepthink: deepThinkEnabled,
        web: webEnabled
      };

      const data = await apiJson("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      removeTypingIndicator();
      addMessage(data.response, "assistant", data.model);

      if (data.chatid) currentChatId = data.chatid;
      if (data.chattitle && currentChatId) upsertChatInSidebar(currentChatId, data.chattitle, true);

      if (data.deepseekremaining != null) {
        const used = 50 - Number(data.deepseekremaining);
        const usage = $("deepseek-usage");
        if (usage) usage.textContent = `${used}/50`;
      }
    } catch (e) {
      removeTypingIndicator();
      addMessage(e.message || "Error connecting to AI.", "error");
    } finally {
      isLoading = false;
    }
  };

  window.useSuggestion = function (text) {
    const input = $("chat-input");
    if (!input) return;
    input.value = text;
    input.focus();
    window.sendMessage();
  };

  // Close dropdowns on outside click (won’t block input)
  document.addEventListener("click", (e) => {
    const menu = $("user-menu");
    const userBtn = e.target.closest(".user-avatar");
    if (menu && menu.style.display === "block" && !userBtn && !menu.contains(e.target)) menu.style.display = "none";

    const selector = $("model-selector");
    const modelBtn = e.target.closest(".model-selector-btn");
    if (selector && selector.style.display === "block" && !modelBtn && !selector.contains(e.target)) selector.style.display = "none";
  });
})();
