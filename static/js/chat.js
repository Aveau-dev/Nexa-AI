/* static/js/chat.js - COMPLETE COMBINED FIXED VERSION (old chat.js + DeepThink/Web + session model persist) */
/* global marked, hljs */
(() => {
  let currentModelKey = (window.NEXAAI && window.NEXAAI.initialModelKey) || "gemini-flash";
  let currentModelName = (window.NEXAAI && window.NEXAAI.initialModelName) || "Gemini 2.5 Flash";

  let currentChatId = null;
  let isLoading = false;

  // NEW
  let deepThinkEnabled = false;
  let webEnabled = false;

  const els = {
    sidebar: () => document.getElementById("sidebar"),
    main: () => document.getElementById("main-content"),
    userMenu: () => document.getElementById("user-menu"),
    modelSelector: () => document.getElementById("model-selector"),
    messages: () => document.getElementById("messages-container"),
    welcome: () => document.getElementById("welcome-section"),
    input: () => document.getElementById("chat-input"),
    modelNameTop: () => document.getElementById("selected-model-name"),
    modelNameFooter: () => document.getElementById("model-info"),
    deepseekUsage: () => document.getElementById("deepseek-usage"),
    chatHistory: () => document.getElementById("chat-history"),

    // NEW
    btnDeepthink: () => document.getElementById("btn-deepthink"),
    btnWeb: () => document.getElementById("btn-web"),
  };

  // ---- UI: sidebar/menu ----
  window.toggleSidebar = function toggleSidebar() {
    const sidebar = els.sidebar();
    const main = els.main();
    if (!sidebar || !main) return;

    if (window.innerWidth < 1024) {
      sidebar.classList.toggle("show");
    } else {
      sidebar.classList.toggle("collapsed");
      main.classList.toggle("sidebar-collapsed");
    }
  };

  window.toggleUserMenu = function toggleUserMenu() {
    const menu = els.userMenu();
    if (!menu) return;
    menu.style.display = (menu.style.display === "block") ? "none" : "block";
  };

  window.toggleModelSelector = function toggleModelSelector() {
    const selector = els.modelSelector();
    if (!selector) return;
    selector.style.display = (selector.style.display === "block") ? "none" : "block";
  };

  // NEW: DeepThink/Web toggles
  window.toggleDeepThink = function toggleDeepThink() {
    deepThinkEnabled = !deepThinkEnabled;
    const b = els.btnDeepthink();
    if (b) {
      b.textContent = deepThinkEnabled ? "DeepThink: On" : "DeepThink: Off";
      b.classList.toggle("active", deepThinkEnabled);
    }
  };

  window.toggleWeb = function toggleWeb() {
    webEnabled = !webEnabled;
    const b = els.btnWeb();
    if (b) {
      b.textContent = webEnabled ? "Web: On" : "Web: Off";
      b.classList.toggle("active", webEnabled);
    }
  };

  // ---- Model selection ----
  window.selectModel = async function selectModel(modelKey, modelName) {
    currentModelKey = modelKey;
    currentModelName = modelName || modelKey;

    if (els.modelNameTop()) els.modelNameTop().textContent = currentModelName;
    if (els.modelNameFooter()) els.modelNameFooter().textContent = currentModelName;

    document.querySelectorAll(".model-card").forEach(card => card.classList.remove("active"));
    const active = document.querySelector(`[data-model="${CSS.escape(modelKey)}"]`);
    if (active) active.classList.add("active");

    // persist in session on backend
    try {
      await fetch("/set-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelKey })
      });
    } catch (e) {
      // ignore
    }

    window.toggleModelSelector();
  };

  // ---- Helpers ----
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

  function addCopyButtons(container) {
    container.querySelectorAll("pre code").forEach((codeBlock) => {
      const pre = codeBlock.parentElement;
      if (!pre || pre.querySelector(".copy-code-btn")) return;

      const btn = document.createElement("button");
      btn.className = "copy-code-btn";
      btn.textContent = "Copy";

      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(codeBlock.textContent);
          btn.textContent = "Copied!";
          setTimeout(() => (btn.textContent = "Copy"), 1200);
        } catch (e) {
          btn.textContent = "Failed";
          setTimeout(() => (btn.textContent = "Copy"), 1200);
        }
      });

      pre.style.position = "relative";
      btn.style.position = "absolute";
      btn.style.top = "8px";
      btn.style.right = "8px";
      pre.appendChild(btn);
    });
  }

  function showTypingIndicator() {
    const container = els.messages();
    if (!container) return;

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
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
  }

  function addMessage(text, role, modelNameForBadge) {
    const container = els.messages();
    if (!container) return;

    const row = document.createElement("div");
    row.className = `message-row ${role}-row`;

    if (role === "user") {
      row.innerHTML = `
        <div class="message user-message">
          <div class="message-content">${escapeHtml(text)}</div>
        </div>
      `;
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
        contentDiv.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));
      }
      addCopyButtons(contentDiv);
    } else {
      row.innerHTML = `
        <div class="message error-message">
          <div class="message-content">${escapeHtml(text)}</div>
        </div>
      `;
    }

    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }

  function setWelcomeVisible(visible) {
    const w = els.welcome();
    if (!w) return;
    w.style.display = visible ? "block" : "none";
  }

  function setActiveChatInSidebar(chatId) {
    document.querySelectorAll(".chat-item").forEach(el => el.classList.remove("active"));
    const active = document.querySelector(`[data-chat-id="${chatId}"]`);
    if (active) active.classList.add("active");
  }

  function upsertChatInSidebar(chatId, title, prepend) {
    const history = els.chatHistory();
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

  // ---- API wrappers ----
  async function apiJson(url, options) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.error || `Request failed (${res.status})`;
      throw new Error(msg);
    }
    return data;
  }

  // ---- Chat CRUD ----
  window.newChat = async function newChat() {
    if (isLoading) return;
    isLoading = true;
    try {
      const data = await apiJson("/chat/new", { method: "POST", headers: { "Content-Type": "application/json" } });
      currentChatId = data.chatid || data.chat_id;

      if (els.messages()) els.messages().innerHTML = "";
      setWelcomeVisible(true);

      upsertChatInSidebar(currentChatId, data.title || "New Chat", true);
      setActiveChatInSidebar(currentChatId);
    } catch (e) {
      addMessage(e.message || "Failed to create new chat.", "error");
    } finally {
      isLoading = false;
    }
  };

  window.loadChat = async function loadChat(chatId) {
    if (isLoading) return;
    isLoading = true;
    try {
      const data = await apiJson(`/chat/${chatId}/messages`, { method: "GET" });
      currentChatId = chatId;

      if (els.messages()) els.messages().innerHTML = "";
      setWelcomeVisible(false);

      (data.messages || []).forEach(m => addMessage(m.content, m.role, m.model));
      setActiveChatInSidebar(chatId);

      if (window.innerWidth < 1024) {
        const sidebar = els.sidebar();
        if (sidebar) sidebar.classList.remove("show");
      }
    } catch (e) {
      addMessage(e.message || "Failed to load chat.", "error");
    } finally {
      isLoading = false;
    }
  };

  window.renameChat = async function renameChat(chatId) {
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

  window.deleteChat = async function deleteChat(chatId) {
    const ok = confirm("Delete this chat? This cannot be undone.");
    if (!ok) return;

    try {
      await apiJson(`/chat/${chatId}/delete`, { method: "DELETE" });

      const item = document.querySelector(`[data-chat-id="${chatId}"]`);
      if (item) item.remove();

      if (currentChatId === chatId) {
        currentChatId = null;
        if (els.messages()) els.messages().innerHTML = "";
        setWelcomeVisible(true);
      }
    } catch (e) {
      addMessage(e.message || "Failed to delete chat.", "error");
    }
  };

  // ---- Send message ----
  window.sendMessage = async function sendMessage() {
    if (isLoading) return;

    const input = els.input();
    const message = (input && input.value ? input.value : "").trim();
    if (!message) return;

    isLoading = true;

    setWelcomeVisible(false);
    addMessage(message, "user");

    input.value = "";
    input.style.height = "auto";

    showTypingIndicator();

    try {
      // NEW: deepthink + web flags added
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

      if (data.image_url) {
        addMessage(data.response || "Image generated.", "assistant", data.model);
        addMessage(data.image_url, "assistant", "Image URL");
      } else {
        addMessage(data.response, "assistant", data.model);
      }

      const newId = data.chatid || data.chat_id;
      if (newId) currentChatId = newId;

      const title = data.chattitle || data.chat_title;
      if (title && currentChatId) upsertChatInSidebar(currentChatId, title, true);

      if (data.deepseekremaining != null && els.deepseekUsage()) {
        const used = 50 - Number(data.deepseekremaining);
        els.deepseekUsage().textContent = `${used}/50`;
      }
    } catch (e) {
      removeTypingIndicator();
      addMessage(e.message || "Error connecting to AI. Please try again.", "error");
    } finally {
      isLoading = false;
    }
  };

  // Suggestions
  window.useSuggestion = function useSuggestion(text) {
    const input = els.input();
    if (!input) return;
    input.value = text;
    input.focus();
    window.sendMessage();
  };

  // Enter key behavior + auto-resize
  function initInput() {
    const input = els.input();
    if (!input) return;

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        window.sendMessage();
      }
    });

    input.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 200) + "px";
    });
  }

  // Close dropdowns on outside click
  function initGlobalClickClose() {
    document.addEventListener("click", (e) => {
      const menu = els.userMenu();
      const userBtn = e.target.closest(".user-avatar");
      if (menu && menu.style.display === "block" && !userBtn && !menu.contains(e.target)) {
        menu.style.display = "none";
      }

      const selector = els.modelSelector();
      const modelBtn = e.target.closest(".model-selector-btn");
      if (selector && selector.style.display === "block" && !modelBtn && !selector.contains(e.target)) {
        selector.style.display = "none";
      }

      if (window.innerWidth < 1024) {
        const sidebar = els.sidebar();
        const toggle = document.getElementById("sidebar-toggle");
        if (sidebar && sidebar.classList.contains("show") && !sidebar.contains(e.target) && toggle && !toggle.contains(e.target)) {
          sidebar.classList.remove("show");
        }
      }
    });
  }

  function initModelLabels() {
    if (els.modelNameTop()) els.modelNameTop().textContent = currentModelName;
    if (els.modelNameFooter()) els.modelNameFooter().textContent = currentModelName;

    document.querySelectorAll(".model-card").forEach(card => card.classList.remove("active"));
    const active = document.querySelector(`[data-model="${CSS.escape(currentModelKey)}"]`);
    if (active) active.classList.add("active");
  }

  // Bootstrap
  initModelLabels();
  initInput();
  initGlobalClickClose();
})();
