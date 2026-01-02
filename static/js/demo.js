/**
 * NexaAI Demo Page JavaScript
 * Handles chat functionality, message rendering, and UI interactions
 */

// Global state
let isResponding = false;

/**
 * Auto-login for demo mode
 * Creates a temporary demo session without requiring sign-up
 */
async function autoLoginDemo() {
  try {
    const res = await fetch("/demo-login", {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    await res.json().catch(() => ({}));
  } catch (e) {
    console.warn("Auto demo login failed", e);
  }
}

/**
 * Initialize the app when DOM is ready
 */
document.addEventListener("DOMContentLoaded", () => {
  // Auto-login for demo
  autoLoginDemo();

  // Setup textarea auto-resize
  const chatInput = document.getElementById('chat-input');
  if (chatInput) {
    chatInput.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    });
  }
});

/**
 * Handle Enter key to send message
 * Shift+Enter for new line
 */
function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

/**
 * Lock/unlock input during AI response
 * @param {boolean} locked - Whether to lock the input
 */
function setInputState(locked) {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('send-btn');
  const sendIcon = document.getElementById('send-icon');
  const stopIcon = document.getElementById('stop-icon');

  isResponding = locked;

  if (input) {
    input.readOnly = locked;
    input.classList.toggle('input-disabled', locked);
  }

  if (btn) {
    btn.disabled = false; // Still clickable to stop
  }

  // Toggle send/stop icon
  if (sendIcon && stopIcon) {
    sendIcon.style.display = locked ? 'none' : 'inline-flex';
    stopIcon.style.display = locked ? 'inline-flex' : 'none';
  }
}

/**
 * Send message to AI
 * Handles the main chat flow
 */
function sendMessage() {
  const input = document.getElementById('chat-input');

  // If responding, stop the request
  if (isResponding) {
    removeTypingIndicator();
    setInputState(false);
    return;
  }

  const message = input.value.trim();
  if (!message) return;

  // Hide welcome section after first message
  const welcome = document.querySelector('.welcome-section');
  if (welcome) welcome.style.display = 'none';

  // Add user message
  addMessage(message, 'user');
  input.value = '';
  input.style.height = 'auto';

  // Show typing indicator and lock input
  showTypingIndicator();
  setInputState(true);

  // Send to backend
  fetch('/demo-chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  })
  .then(res => res.json())
  .then(data => {
    removeTypingIndicator();
    setInputState(false);

    if (data.response) {
      addMessage(data.response, 'assistant');
    } else if (data.error) {
      addMessage(data.error, 'error');
    }
  })
  .catch(() => {
    removeTypingIndicator();
    setInputState(false);
    addMessage('Error connecting to AI. Please try again.', 'error');
  });
}

/**
 * Format message with Markdown and syntax highlighting
 * @param {string} text - Raw text to format
 * @returns {string} Formatted HTML
 */
function formatDemoMessage(text) {
  // Fallback if marked.js is not loaded
  if (typeof marked === 'undefined') {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
  }

  // Configure marked for code highlighting
  marked.setOptions({
    highlight(code, lang) {
      if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return (typeof hljs !== 'undefined') ? hljs.highlightAuto(code).value : code;
    },
    breaks: true,
    gfm: true
  });

  return marked.parse(text);
}

/**
 * Add copy buttons to code blocks
 * @param {HTMLElement} container - Container with code blocks
 */
function addDemoCopyButtons(container) {
  container.querySelectorAll('pre code').forEach((codeBlock) => {
    const pre = codeBlock.parentElement;
    const btn = document.createElement('button');
    btn.className = 'copy-code-btn';
    btn.textContent = 'Copy';

    btn.addEventListener('click', () => {
      navigator.clipboard.writeText(codeBlock.textContent);
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });

    pre.style.position = 'relative';
    btn.style.position = 'absolute';
    btn.style.top = '8px';
    btn.style.right = '8px';
    pre.appendChild(btn);
  });
}

/**
 * Add a message to the chat
 * @param {string} text - Message content
 * @param {string} role - 'user', 'assistant', or 'error'
 */
function addMessage(text, role) {
  const container = document.getElementById('messages-container');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message-row ${role}-row`;

  if (role === 'user') {
    // User message bubble
    messageDiv.innerHTML = `
      <div class="message user-message">
        <div class="message-content">${escapeHtml(text)}</div>
      </div>`;

  } else if (role === 'assistant') {
    // Assistant message with avatar
    messageDiv.innerHTML = `
      <div class="message assistant-message">
        <div class="avatar">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
            <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
          </svg>
        </div>
        <div class="message-content markdown-content"></div>
      </div>`;

  } else {
    // Error message
    messageDiv.innerHTML = `
      <div class="message error-message">
        <div class="message-content">⚠️ ${escapeHtml(text)}</div>
      </div>`;
  }

  // Format assistant messages with Markdown
  if (role === 'assistant') {
    const contentDiv = messageDiv.querySelector('.markdown-content');
    contentDiv.innerHTML = formatDemoMessage(text);

    // Apply syntax highlighting
    if (typeof hljs !== 'undefined') {
      contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
    }

    // Add copy buttons to code blocks
    addDemoCopyButtons(contentDiv);

    // Add action buttons (Listen, Like, Dislike, Share)
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
      <button class="msg-btn" data-action="share" title="Share">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="18" cy="5" r="3"></circle>
          <circle cx="6" cy="12" r="3"></circle>
          <circle cx="18" cy="19" r="3"></circle>
          <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
          <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
        </svg>
      </button>
    `;

    // Handle action button clicks
    actions.addEventListener('click', (e) => {
      const btn = e.target.closest('.msg-btn');
      if (btn) handleMessageAction(btn.dataset.action, text, actions);
    });

    contentDiv.appendChild(actions);
  }

  container.appendChild(messageDiv);
  container.scrollTop = container.scrollHeight;
}

/**
 * Handle message action button clicks
 * @param {string} action - 'listen', 'like', 'dislike', or 'share'
 * @param {string} text - Message text
 * @param {HTMLElement} actionsEl - Actions container element
 */
function handleMessageAction(action, text, actionsEl) {
  if (action === 'listen') {
    // Text-to-speech
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

  } else if (action === 'share') {
    // Use Web Share API if available, otherwise copy to clipboard
    if (navigator.share) {
      navigator.share({ text });
    } else {
      navigator.clipboard.writeText(text);
      alert('Message copied to clipboard');
    }
  }
}

/**
 * Open Terms & Policies modal
 */
function openTermsModal() {
  const modal = document.getElementById('terms-modal');
  if (!modal) return;

  modal.style.display = 'flex';
  document.documentElement.style.overflow = 'hidden';
  document.body.style.overflow = 'hidden';
}

/**
 * Close Terms & Policies modal
 */
function closeTermsModal() {
  const modal = document.getElementById('terms-modal');
  if (!modal) return;

  modal.style.display = 'none';
  document.documentElement.style.overflow = '';
  document.body.style.overflow = '';
}

/**
 * Close modal when clicking outside
 */
document.addEventListener('click', (e) => {
  const modal = document.getElementById('terms-modal');
  if (modal && e.target === modal) {
    closeTermsModal();
  }
});

/**
 * Show typing indicator (animated dots)
 */
function showTypingIndicator() {
  const container = document.getElementById('messages-container');
  const existing = document.getElementById('typing-indicator');
  if (existing) existing.remove();

  const typingDiv = document.createElement('div');
  typingDiv.className = 'message-row assistant-row typing-indicator';
  typingDiv.id = 'typing-indicator';
  typingDiv.innerHTML = `
    <div class="message assistant-message">
      <div class="avatar">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
          <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
        </svg>
      </div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;

  container.appendChild(typingDiv);
  container.scrollTop = container.scrollHeight;
}

/**
 * Remove typing indicator
 */
function removeTypingIndicator() {
  const indicator = document.getElementById('typing-indicator');
  if (indicator) indicator.remove();
}

/**
 * Escape HTML to prevent XSS
 * @param {string} text - Text to escape
 * @returns {string} Escaped HTML
 */




// Detects Gemini API errors
const isQuotaError = errorMsg.includes('quota') || 
                     errorMsg.includes('rate limit') || 
                     errorMsg.includes('resource exhausted') ||
                     errorMsg.includes('429') ||
                     errorMsg.includes('limit exceeded');

// Auto-retry with shortened message
if (isQuotaError && retryCount < MAX_RETRIES) {
  const shortenedMsg = shortenMessage(message);
  sendMessage(shortenedMsg, true);
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}
