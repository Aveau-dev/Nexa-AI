/* static/js/chat.js - Complete with Stop/Abort functionality */
window.Chat = (function () {
  let currentModelKey = (window.NEXAAI && window.NEXAAI.initialModelKey) || "gemini-flash";
  let currentModelName = (window.NEXAAI && window.NEXAAI.initialModelName) || "Gemini 2.5 Flash";

  let currentChatId = null;
  let isLoading = false;
  let abortController = null; // For stopping AI responses

  let deepThinkEnabled = false;
  let webEnabled = false;

  let deepThinkTimer = null;
  let deepThinkStart = 0;

  function $(id) { return document.getElementById(id); }

  // -----------------------------
  // Theme (data-theme on <html>)
  // -----------------------------
  function getSavedTheme() {
    return localStorage.getItem("nexa_theme") || "dark";
  }

  function applyTheme(theme) {
    const t = (theme === "light") ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("nexa_theme", t);

    const btn = $("btn-theme");
    if (btn) btn.textContent = (t === "light") ? "Theme: Light" : "Theme: Dark";
  }

  window.toggleTheme = function () {
    const curr = document.documentElement.getAttribute("data-theme") || getSavedTheme();
    applyTheme(curr === "light" ? "dark" : "light");
  };

  // Apply theme on load
  applyTheme(getSavedTheme());

  // ---- DeepThink timer UI ----
  function deepThinkStartTimer() {
    const box = $("deepthink-status");
    const sec = $("deepthink-seconds");
    if (!box || !sec) return;

    deepThinkStart = Date.now();
    box.style.display = "block";
    sec.textContent = "0s";

    clearInterval(deepThinkTimer);
    deepThinkTimer = setInterval(() => {
      const s = Math.floor((Date.now() - deepThinkStart) / 1000);
      sec.textContent = `${s}s`;
    }, 250);
  }

  function deepThinkStopTimer() {
    clearInterval(deepThinkTimer);
    deepThinkTimer = null;

    const box = $("deepthink-status");
    if (box) box.style.display = "none";
  }

  // ---- Stop AI Response (Perplexity-style) ----
  function stopAIResponse() {
    if (!isLoading) return;

    console.log('User stopped AI response');

    // Cancel the fetch request
    if (abortController) {
      abortController.abort();
    }

    hideTypingIndicator();
    deepThinkStopTimer();
    addMessage('⏸️ Response stopped by user', 'system');

    // Reset state
    setInputState(false);
    isLoading = false;
    abortController = null;
  }

  // ---- Input state (lock/unlock) ----
  function setInputState(locked) {
    const input = $("chat-input");
    const btn = $("send-btn");
    const sendIcon = $("send-icon");
    const stopIcon = $("stop-icon");

    isLoading = locked;

    if (input) {
      input.readOnly = locked;
      input.classList.toggle('input-disabled', locked);
    }

    if (btn) {
      btn.disabled = false; // Always clickable for stop
    }

    // Toggle send/stop icon
    if (sendIcon && stopIcon) {
      sendIcon.style.display = locked ? 'none' : 'inline-flex';
      stopIcon.style.display = locked ? 'inline-flex' : 'none';
    }
  }

  // -----------------------------
  // Bindings (run after chat view loads)
  // -----------------------------
  function bindChatViewOnce() {
    const dt = $("btn-deepthink");
    const wb = $("btn-web");
    const sendBtn = $("send-btn");
    const input = $("chat-input");

    // DeepThink pill
    if (dt && !dt.dataset.bound) {
      dt.dataset.bound = "1";
      dt.addEventListener("click", () => {
        deepThinkEnabled = !deepThinkEnabled;
        dt.classList.toggle("active", deepThinkEnabled);
      });
    }

    // Web Search pill
    if (wb && !wb.dataset.bound) {
      wb.dataset.bound = "1";
      wb.addEventListener("click", () => {
        webEnabled = !webEnabled;
        wb.classList.toggle("active", webEnabled);
      });
    }

    // Send button (also handles stop)
    if (sendBtn && !sendBtn.dataset.bound) {
      sendBtn.dataset.bound = "1";
      sendBtn.addEventListener("click", () => {
        if (isLoading) {
          stopAIResponse();
        } else {
          sendMessage();
        }
      });
    }

    // Textarea
    if (input && !input.dataset.bound) {
      input.dataset.bound = "1";

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          if (!isLoading) sendMessage();
        }
      });

      input.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = Math.min(this.scrollHeight, 200) + "px";
      });

      input.removeAttribute("disabled");
    }

    const top = $("selected-model-name");
    const foot = $("model-info");
    if (top) top.textContent = currentModelName;
    if (foot) foot.textContent = currentModelName;
  }

  // Called by Router when chat view loads
  function onViewLoaded() {
    bindChatViewOnce();
  }

  // ---- global UI funcs used by dashboard.html ----
  window.toggleSidebar = function () {
    const sidebar = $("sidebar");
    const main = $("main-content");
    if (!sidebar || !main) return;

    if (window.innerWidth < 1024) {
      sidebar.classList.toggle("show");
    } else {
      sidebar.classList.toggle("collapsed");
      main.classList.toggle("sidebar-collapsed");
    }
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
    if (typeof marked === "undefined") {
      return escapeHtml(text).replace(/\n/g, "<br>");
    }
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

  function setWelcomeVisible(visible) {
    const w = $("welcome-section");
    if (w) w.style.display = visible ? "block" : "none";
  }

  function addMessage(text, role, modelNameForBadge, reasoningText = null, meta = null) {
    const container = $("messages-container");
    if (!container) return;

    setWelcomeVisible(false);

    const row = document.createElement("div");
    row.className = `message-row ${role}-row`;

    if (role === "user") {
      row.innerHTML = `
        <div class="message user-message">
          <div class="message-content">${escapeHtml(text)}</div>
        </div>`;

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
            <div class="assistant-meta"></div>
            ${reasoningText ? `
              <details class="reasoning-panel" style="margin:.35rem 0 .5rem 0;">
                <summary style="cursor:pointer; color:var(--text-secondary);">DeepThink (summary)</summary>
                <div style="margin-top:.35rem; color:var(--text-primary);">${escapeHtml(reasoningText)}</div>
              </details>` : ``}
            <div class="markdown-content"></div>
          </div>
        </div>
      `;

      const metaEl = row.querySelector(".assistant-meta");
      if (metaEl && meta?.thoughtSeconds) {
        const s = Number(meta.thoughtSeconds).toFixed(1);
        metaEl.textContent = meta.webEnabled
          ? `Thought for ${s}s • Web search on`
          : `Thought for ${s}s`;
        metaEl.style.cssText = "color: var(--text-secondary); font-size: 0.8rem; margin: 0.25rem 0 0.5rem 0;";
      } else if (metaEl) {
        metaEl.remove();
      }

      const contentDiv = row.querySelector(".markdown-content");
      contentDiv.innerHTML = formatMessage(text);

      if (typeof hljs !== "undefined") {
        contentDiv.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
      }

      // Add action buttons (like, listen, share)
      addMessageActions(contentDiv, text);

    } else if (role === "system") {
      // System message (stop notification)
      row.innerHTML = `
        <div class="message system-message">
          <div class="message-content">${escapeHtml(text)}</div>
        </div>`;

    } else {
      // Error message
      row.innerHTML = `
        <div class="message error-message">
          <div class="message-content">${escapeHtml(text)}</div>
        </div>`;
    }

    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }

  // ---- Add action buttons to assistant messages ----
  function addMessageActions(contentDiv, text) {
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.innerHTML = `
      <button class="msg-btn" data-action="listen" title="Listen">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
      </button>
      <button class="msg-btn" data-action="like" title="Like">👍</button>
      <button class="msg-btn" data-action="dislike" title="Dislike">👎</button>
      <button class="msg-btn" data-action="copy" title="Copy">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
      </button>
    `;

    actions.addEventListener('click', (e) => {
      const btn = e.target.closest('.msg-btn');
      if (btn) handleMessageAction(btn.dataset.action, text, actions);
    });

    contentDiv.appendChild(actions);
  }

  // ---- Handle message action buttons ----
  function handleMessageAction(action, text, actionsEl) {
    if (action === 'listen') {
      try {
        const utter = new SpeechSynthesisUtterance(text);
        speechSynthesis.speak(utter);
      } catch (e) {
        console.warn('Text-to-speech not supported', e);
      }

    } else if (action === 'like') {
      actionsEl.classList.add('liked');
      actionsEl.classList.remove('disliked');

    } else if (action === 'dislike') {
      actionsEl.classList.add('disliked');
      actionsEl.classList.remove('liked');

    } else if (action === 'copy') {
      navigator.clipboard.writeText(text).then(() => {
        const btn = actionsEl.querySelector('[data-action="copy"]');
        if (btn) {
          btn.textContent = '✓';
          setTimeout(() => {
            btn.innerHTML = `
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
            `;
          }, 1500);
        }
      });
    }
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

  function hideTypingIndicator() {
    const t = $("typing-indicator");
    if (t) t.remove();
  }

  // ---- Chat actions ----
  window.newChat = async function () {
    currentChatId = null;
    const container = $("messages-container");
    if (container) container.innerHTML = "";
    setWelcomeVisible(true);

    document.querySelectorAll(".chat-item").forEach(c => c.classList.remove("active"));

    try {
      const res = await fetch("/new-chat", { method: "POST" });
      const data = await res.json();
      if (data.chat_id) {
        currentChatId = data.chat_id;
        await refreshChatHistory();
      }
    } catch (err) {
      console.error("newChat error:", err);
    }
  };

  window.loadChat = async function (chatId) {
    currentChatId = chatId;
    const container = $("messages-container");
    if (container) container.innerHTML = "";
    setWelcomeVisible(false);

    document.querySelectorAll(".chat-item").forEach(c => {
      c.classList.toggle("active", c.dataset.chatId === chatId);
    });

    try {
      const res = await fetch(`/get-chat/${chatId}`);
      const data = await res.json();

      if (data.messages && data.messages.length > 0) {
        data.messages.forEach(msg => {
          addMessage(
            msg.content,
            msg.role,
            msg.role === "assistant" ? currentModelName : null,
            msg.reasoning_content || null,
            msg.meta || null
          );
        });
      }
    } catch (err) {
      console.error("loadChat error:", err);
      addMessage("Failed to load chat.", "error");
    }
  };

  window.renameChat = async function (chatId) {
    const newTitle = prompt("Enter new chat title:");
    if (!newTitle || !newTitle.trim()) return;

    try {
      await fetch(`/rename-chat/${chatId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle.trim() })
      });
      await refreshChatHistory();
    } catch (err) {
      console.error("renameChat error:", err);
    }
  };

  window.deleteChat = async function (chatId) {
    if (!confirm("Delete this chat?")) return;

    try {
      await fetch(`/delete-chat/${chatId}`, { method: "POST" });

      if (currentChatId === chatId) {
        await window.newChat();
      } else {
        await refreshChatHistory();
      }
    } catch (err) {
      console.error("deleteChat error:", err);
    }
  };

  async function refreshChatHistory() {
    try {
      const res = await fetch("/get-chats");
      const data = await res.json();

      const historyDiv = $("chat-history");
      if (!historyDiv) return;

      if (!data.chats || data.chats.length === 0) {
        historyDiv.innerHTML = '<div class="empty-state">No chats yet. Start a new conversation!</div>';
        return;
      }

      historyDiv.innerHTML = data.chats.map(chat => `
        <div class="chat-item ${chat.id === currentChatId ? 'active' : ''}"
             data-chat-id="${chat.id}"
             onclick="window.loadChat('${chat.id}')">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"></path>
          </svg>
          <span class="chat-title">${escapeHtml(chat.title)}</span>
          <div class="chat-actions">
            <button class="icon-btn" title="Rename"
                    onclick="event.stopPropagation(); window.renameChat('${chat.id}')">✏️</button>
            <button class="icon-btn" title="Delete"
                    onclick="event.stopPropagation(); window.deleteChat('${chat.id}')">🗑️</button>
          </div>
        </div>
      `).join("");
    } catch (err) {
      console.error("refreshChatHistory error:", err);
    }
  }

  // ---- Send message with abort support ----
  async function sendMessage() {
    const input = $("chat-input");
    if (!input || isLoading) return;

    const text = input.value.trim();
    if (!text) return;

    input.value = "";
    input.style.height = "auto";

    if (!currentChatId) {
      const res = await fetch("/new-chat", { method: "POST" });
      const data = await res.json();
      if (data.chat_id) {
        currentChatId = data.chat_id;
        await refreshChatHistory();
      }
    }

    addMessage(text, "user");
    showTypingIndicator();
    setInputState(true);

    if (deepThinkEnabled) {
      deepThinkStartTimer();
    }

    // Create new AbortController for this request
    abortController = new AbortController();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          chat_id: currentChatId,
          model: currentModelKey,
          deepthink: deepThinkEnabled,
          web: webEnabled
        }),
        signal: abortController.signal // Add abort signal
      });

      const data = await res.json();

      hideTypingIndicator();
      deepThinkStopTimer();
      setInputState(false);
      abortController = null;

      if (data.error) {
        addMessage(data.error, "error");
      } else {
        addMessage(
          data.response,
          "assistant",
          currentModelName,
          data.reasoning_content || null,
          data.meta || null
        );

        if (data.deepseek_count !== undefined) {
          const usage = $("deepseek-usage");
          if (usage) usage.textContent = data.deepseek_count;
        }

        await refreshChatHistory();
      }
    } catch (err) {
      // Check if request was aborted by user
      if (err.name === 'AbortError') {
        console.log('Request aborted by user');
        return; // Don't show error - we already showed stop message
      }

      hideTypingIndicator();
      deepThinkStopTimer();
      setInputState(false);
      abortController = null;
      console.error("sendMessage error:", err);
      addMessage("Network error. Please try again.", "error");
    }
  }

  window.useSuggestion = function (text) {
    const input = $("chat-input");
    if (input) {
      input.value = text;
      input.focus();
    }
  };

  // ---- Export public API for Router ----
  return {
    onViewLoaded
  };
})();
