/* global window, document, fetch */
(function () {
  const Chat = {
    init() {
      this.currentChatId = null;
      this.isLoading = false;
    },

    onViewMounted() {
      // bind chat view controls if present
      const sendBtn = document.getElementById('chat-send-btn');
      const input = document.getElementById('chat-input');
      const newBtn = document.getElementById('chat-new-btn');

      if (newBtn) newBtn.addEventListener('click', () => this.newChatAndOpen());

      if (sendBtn) sendBtn.addEventListener('click', () => this.sendMessage());
      if (input) {
        input.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
          }
        });
        input.addEventListener('input', () => window.UI && UI.autoResize(input));
      }
    },

    async refreshSidebar() {
      const list = document.getElementById('chat-history');
      if (!list) return;

      list.innerHTML = `<div class="empty-state">Loading chats…</div>`;

      try {
        const res = await fetch('/api/chats');
        const data = await res.json();

        if (!res.ok) {
          list.innerHTML = `<div class="empty-state">${data.error || 'Failed to load chats'}</div>`;
          return;
        }

        if (!data.chats || data.chats.length === 0) {
          list.innerHTML = `<div class="empty-state">No chats yet. Start a new conversation!</div>`;
          return;
        }

        list.innerHTML = '';
        for (const c of data.chats) {
          const item = document.createElement('div');
          item.className = 'chat-item';
          item.dataset.chatId = c.id;
          item.innerHTML = `
            <span class="chat-title">${this.escapeHtml(c.title || 'New Chat')}</span>
          `;
          item.addEventListener('click', () => this.openChat(c.id));
          list.appendChild(item);
        }
      } catch (e) {
        console.error(e);
        list.innerHTML = `<div class="empty-state">Network error loading chats</div>`;
      }
    },

    async newChatAndOpen() {
      try {
        const res = await fetch('/chat/new', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || !data.success) return alert(data.error || 'Failed to create chat');

        this.currentChatId = data.chat_id;
        await this.refreshSidebar();
        await this.openChat(this.currentChatId);
      } catch (e) {
        console.error(e);
        alert('Network error creating chat');
      }
    },

    async openChat(chatId) {
      const messages = document.getElementById('messages');
      if (!messages) return;

      this.currentChatId = chatId;
      messages.innerHTML = `<div class="loading">Loading messages…</div>`;

      try {
        const res = await fetch(`/chat/${chatId}/messages`);
        const data = await res.json();
        if (!res.ok) return alert(data.error || 'Failed to load messages');

        messages.innerHTML = '';
        for (const m of data.messages || []) {
          this.addMessage(m.role, m.content, m.model);
        }
        messages.scrollTop = messages.scrollHeight;
      } catch (e) {
        console.error(e);
        messages.innerHTML = `<div class="empty-state">Network error</div>`;
      }
    },

    async sendMessage() {
      const input = document.getElementById('chat-input');
      const messages = document.getElementById('messages');
      if (!input || !messages) return;

      const text = (input.value || '').trim();
      if (!text || this.isLoading) return;

      this.isLoading = true;
      input.value = '';
      if (window.UI) UI.autoResize(input);

      this.addMessage('user', text, null);
      this.addTyping();

      try {
        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            model: window.Models ? Models.getSelected() : 'gemini-flash',
            chat_id: this.currentChatId
          })
        });

        const data = await res.json();
        this.removeTyping();

        if (!res.ok || data.error) {
          this.addMessage('error', data.error || 'Chat failed', null);
          this.isLoading = false;
          return;
        }

        // server may create chat if none
        this.currentChatId = data.chat_id;
        this.addMessage('assistant', data.response, data.model);

        await this.refreshSidebar();
      } catch (e) {
        console.error(e);
        this.removeTyping();
        this.addMessage('error', 'Network error talking to server.', null);
      } finally {
        this.isLoading = false;
      }
    },

    addMessage(role, text, modelName) {
      const messages = document.getElementById('messages');
      if (!messages) return;

      const row = document.createElement('div');
      row.className = `message-row ${role}-row`;

      const safe = this.escapeHtml(text);

      if (role === 'assistant') {
        const rendered = (window.marked && typeof window.marked.parse === 'function')
          ? window.marked.parse(text)
          : safe.replace(/\n/g, '<br>');

        row.innerHTML = `
          <div class="message assistant-message">
            <div class="message-content">
              ${modelName ? `<div class="model-badge">${this.escapeHtml(modelName)}</div>` : ''}
              <div class="message-text markdown-content">${rendered}</div>
            </div>
          </div>
        `;

        // highlight code
        setTimeout(() => {
          row.querySelectorAll('pre code').forEach(block => {
            if (window.hljs) window.hljs.highlightElement(block);
          });
        }, 0);
      } else if (role === 'user') {
        row.innerHTML = `
          <div class="message user-message">
            <div class="message-content">
              <div class="message-text">${safe}</div>
            </div>
          </div>
        `;
      } else {
        row.innerHTML = `
          <div class="message error-message">
            <div class="message-content">${safe}</div>
          </div>
        `;
      }

      messages.appendChild(row);
      messages.scrollTop = messages.scrollHeight;
    },

    addTyping() {
      const messages = document.getElementById('messages');
      if (!messages) return;
      this.removeTyping();

      const row = document.createElement('div');
      row.id = 'typing-indicator';
      row.className = 'message-row assistant-row typing-indicator';
      row.innerHTML = `<div class="loading">Thinking…</div>`;
      messages.appendChild(row);
      messages.scrollTop = messages.scrollHeight;
    },

    removeTyping() {
      const t = document.getElementById('typing-indicator');
      if (t) t.remove();
    },

    escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = s ?? '';
      return d.innerHTML;
    }
  };

  window.Chat = Chat;
})();
