'use strict';

// ═══════════════════════════════════════════════════════════════════
// STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════
let isResponding = false;
let abortController = null;
let lastUserMessage = '';
let currentFile = null;
let currentFileType = null;
let imageGenMode = false;

// ═══════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════
async function autoLoginDemo() {
    try {
        const res = await fetch("/demo-login", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        await res.json().catch(() => ({}));
        console.log('✅ Demo session initialized');
    } catch (e) {
        console.warn("Auto demo login failed", e);
    }
}

document.addEventListener("DOMContentLoaded", async () => {
    await autoLoginDemo();
    
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
        chatInput.addEventListener('keydown', handleKeyDown);
    }
    
    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) {
        sendBtn.addEventListener('click', handleSendClick);
    }
    
    // File upload handlers
    const attachBtn = document.getElementById('attach-btn');
    const fileInput = document.getElementById('file-input');
    
    if (attachBtn && fileInput) {
        attachBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Image generation button
    const imageGenBtn = document.getElementById('image-gen-btn');
    if (imageGenBtn) {
        imageGenBtn.addEventListener('click', toggleImageGenMode);
    }
    
    console.log('🚀 NexaAI Demo - Enhanced mode with Vision + Image Gen + Files');
});

// ═══════════════════════════════════════════════════════════════════
// FILE HANDLING
// ═══════════════════════════════════════════════════════════════════
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    currentFile = file;
    currentFileType = file.type;
    
    // Show file preview
    const uploadArea = document.getElementById('file-upload-area');
    const preview = document.getElementById('file-preview');
    
    const fileIcon = currentFileType.startsWith('image/') ? '🖼️' : '📄';
    const fileName = file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name;
    
    preview.innerHTML = `
        <span class="file-preview-icon">${fileIcon}</span>
        <span class="file-preview-name">${fileName}</span>
        <button class="file-remove-btn" onclick="clearFile()">✕</button>
    `;
    
    uploadArea.classList.add('active');
    
    // If it's an image, prepare for vision
    if (currentFileType.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            currentFile = e.target.result; // Base64 data
        };
        reader.readAsDataURL(file);
    }
}

function clearFile() {
    currentFile = null;
    currentFileType = null;
    const uploadArea = document.getElementById('file-upload-area');
    const fileInput = document.getElementById('file-input');
    uploadArea.classList.remove('active');
    fileInput.value = '';
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE GENERATION MODE
// ═══════════════════════════════════════════════════════════════════
function toggleImageGenMode() {
    imageGenMode = !imageGenMode;
    const btn = document.getElementById('image-gen-btn');
    const input = document.getElementById('chat-input');
    
    if (imageGenMode) {
        btn.style.background = 'var(--color-primary)';
        btn.style.color = 'var(--color-btn-primary-text)';
        input.placeholder = 'Describe the image you want to generate...';
    } else {
        btn.style.background = 'transparent';
        btn.style.color = 'var(--color-text-secondary)';
        input.placeholder = 'Ask anything...';
    }
}

// ═══════════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════════
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSendClick();
    }
}

function handleSendClick() {
    if (isResponding) {
        stopAIResponse();
    } else {
        sendMessage();
    }
}

// ═══════════════════════════════════════════════════════════════════
// INPUT STATE
// ═══════════════════════════════════════════════════════════════════
function setInputState(locked) {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('send-btn');
    const sendIcon = document.getElementById('send-icon');
    const stopIcon = document.getElementById('stop-icon');
    
    isResponding = locked;
    
    if (input) {
        input.readOnly = locked;
        input.classList.toggle('input-disabled', locked);
        if (!locked) input.focus();
    }
    
    if (btn) {
        btn.disabled = false;
        btn.title = locked ? 'Stop response' : 'Send message';
    }
    
    if (sendIcon && stopIcon) {
        sendIcon.style.display = locked ? 'none' : 'inline-flex';
        stopIcon.style.display = locked ? 'inline-flex' : 'none';
    }
}

// ═══════════════════════════════════════════════════════════════════
// STOP RESPONSE
// ═══════════════════════════════════════════════════════════════════
function stopAIResponse() {
    if (!isResponding) return;
    
    console.log('🛑 Stopped by user');
    if (abortController) {
        abortController.abort();
    }
    removeTypingIndicator();
    addMessageUI('⏸️ Response stopped by user', 'system');
    setInputState(false);
    isResponding = false;
    abortController = null;
}

// ═══════════════════════════════════════════════════════════════════
// SEND MESSAGE (ENHANCED WITH IMAGE GEN & VISION)
// ═══════════════════════════════════════════════════════════════════
async function sendMessage(retryMessage = null) {
    const input = document.getElementById('chat-input');
    const message = retryMessage || (input ? input.value.trim() : '');
    
    if (!message && !currentFile) return;
    
    lastUserMessage = message;
    
    const welcome = document.getElementById('welcome-section');
    if (welcome) {
        welcome.style.display = 'none';
    }
    
    // Handle Image Generation Mode
    if (imageGenMode && message) {
        addMessageUI(message, 'user');
        if (input && !retryMessage) {
            input.value = '';
            input.style.height = 'auto';
        }
        imageGenMode = false;
        toggleImageGenMode(); // Reset button
        await generateImage(message);
        return;
    }
    
    // Display user message with file indicator
    if (currentFile && currentFileType.startsWith('image/')) {
        addMessageUI(message + '\n\n🖼️ *Image attached for analysis*', 'user');
    } else if (currentFile) {
        addMessageUI(message + '\n\n📄 *File attached for analysis*', 'user');
    } else {
        addMessageUI(message, 'user');
    }
    
    if (input && !retryMessage) {
        input.value = '';
        input.style.height = 'auto';
    }
    
    showTypingIndicator();
    setInputState(true);
    
    abortController = new AbortController();
    
    try {
        const payload = {
            message: message,
            model: 'google/gemini-2.0-flash-exp:free' // Vision-enabled model
        };
        
        // Add image data if present
        if (currentFile && currentFileType.startsWith('image/')) {
            payload.image = currentFile.split(',')[1]; // Get base64 part
        }
        
        const res = await fetch('/demo-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: abortController.signal
        });
        
        clearFile(); // Clear file after sending
        
        if (res.ok && res.headers.get('content-type')?.includes('text/event-stream')) {
            await handleSSEStream(res);
        } else {
            const data = await res.json();
            removeTypingIndicator();
            setInputState(false);
            abortController = null;
            
            if (data.response) {
                addMessageUI(data.response, 'assistant');
            } else if (data.error) {
                handleError(data.error, data.retryable);
            }
        }
    } catch (err) {
        if (err.name === 'AbortError') return;
        
        removeTypingIndicator();
        setInputState(false);
        abortController = null;
        console.error('Fetch error:', err);
        handleError('Network error. Please check your connection.', true);
    }
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════════
async function generateImage(prompt) {
    showTypingIndicator();
    setInputState(true);
    
    try {
        // Using Pollinations AI (free)
        const cleanPrompt = encodeURIComponent(prompt);
        const seed = Math.floor(Math.random() * 10000);
        const imageUrl = `https://image.pollinations.ai/prompt/${cleanPrompt}?width=1024&height=1024&nologo=true&seed=${seed}`;
        
        removeTypingIndicator();
        
        // Display generated image
        addMessageUI(`
            <p>I've generated an image based on your prompt:</p>
            <img src="${imageUrl}" alt="Generated image" class="message-image" onclick="window.open('${imageUrl}', '_blank')">
            <p><small>Click image to view full size · Powered by Pollinations AI</small></p>
        `, 'assistant');
        
        setInputState(false);
    } catch (error) {
        removeTypingIndicator();
        handleError('Failed to generate image. Please try again.', true);
        setInputState(false);
    }
}

// ═══════════════════════════════════════════════════════════════════
// SSE STREAMING (ENHANCED WITH REASONING)
// ═══════════════════════════════════════════════════════════════════
async function handleSSEStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullResponse = '';
    let reasoning = '';
    let messageElement = null;
    let hasCreatedMessage = false;
    let isReasoningMode = false;
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (!line.trim() || line.startsWith(':')) continue;
                
                if (line.startsWith('event:')) {
                    const event = line.substring(6).trim();
                    if (event === 'done') {
                        removeTypingIndicator();
                        setInputState(false);
                        abortController = null;
                        return;
                    }
                } else if (line.startsWith('data:')) {
                    const jsonStr = line.substring(5).trim();
                    if (!jsonStr || jsonStr === '{}') continue;
                    
                    try {
                        const data = JSON.parse(jsonStr);
                        
                        if (data.messages_analyzed !== undefined && !hasCreatedMessage) {
                            removeTypingIndicator();
                            messageElement = createStreamingMessage();
                            hasCreatedMessage = true;
                        } else if (data.delta && messageElement) {
                            // Check for reasoning markers
                            if (data.delta.includes('<think>') || data.delta.includes('<reasoning>')) {
                                isReasoningMode = true;
                            }
                            
                            if (isReasoningMode) {
                                reasoning += data.delta;
                                if (data.delta.includes('</think>') || data.delta.includes('</reasoning>')) {
                                    isReasoningMode = false;
                                }
                            } else {
                                fullResponse += data.delta;
                                updateStreamingMessage(messageElement, fullResponse, reasoning);
                            }
                        } else if (data.error) {
                            removeTypingIndicator();
                            handleError(data.error, data.retryable !== false);
                            setInputState(false);
                            abortController = null;
                            return;
                        }
                    } catch (e) {
                        console.warn('JSON parse error:', e);
                    }
                }
            }
        }
        
        removeTypingIndicator();
        if (fullResponse && !hasCreatedMessage) {
            addMessageUI(fullResponse, 'assistant', reasoning);
        }
        setInputState(false);
        abortController = null;
    } catch (error) {
        console.error('Stream error:', error);
        removeTypingIndicator();
        setInputState(false);
        abortController = null;
    }
}

// ═══════════════════════════════════════════════════════════════════
// UI FUNCTIONS (ENHANCED)
// ═══════════════════════════════════════════════════════════════════
function createStreamingMessage() {
    const container = document.getElementById('messages-container');
    if (!container) return null;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-row assistant-row streaming-message';
    messageDiv.innerHTML = `
        <div class="message assistant-message">
            <div class="avatar">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
                </svg>
            </div>
            <div class="message-content">
                <div class="markdown-content"></div>
            </div>
        </div>
    `;
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
    
    return messageDiv.querySelector('.markdown-content');
}

function updateStreamingMessage(element, text, reasoning = '') {
    if (!element) return;
    
    let html = '';
    
    // Add reasoning block if present
    if (reasoning) {
        const cleanReasoning = reasoning
            .replace(/<think>|<reasoning>/gi, '')
            .replace(/<\/think>|<\/reasoning>/gi, '');
        
        html += `
            <div class="reasoning-block">
                <div class="reasoning-header">
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <path d="M12 16v-4m0-4h.01"></path>
                    </svg>
                    <span>Reasoning</span>
                </div>
                <div class="reasoning-content">${escapeHtml(cleanReasoning)}</div>
            </div>
        `;
    }
    
    html += formatDemoMessage(text);
    element.innerHTML = html;
    
    if (typeof hljs !== 'undefined') {
        element.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
    }
    
    addDemoCopyButtons(element);
    
    const container = document.getElementById('messages-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

function addMessageUI(text, role, reasoning = '') {
    const container = document.getElementById('messages-container');
    if (!container) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message-row ${role}-row`;
    
    if (role === 'user') {
        messageDiv.innerHTML = `
            <div class="message user-message">
                <div class="message-content">${escapeHtml(text).replace(/\n/g, '<br>')}</div>
            </div>
        `;
    } else if (role === 'assistant') {
        messageDiv.innerHTML = `
            <div class="message assistant-message">
                <div class="avatar">
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                        <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"></path>
                    </svg>
                </div>
                <div class="message-content">
                    <div class="markdown-content"></div>
                </div>
            </div>
        `;
        
        const contentDiv = messageDiv.querySelector('.markdown-content');
        
        let html = '';
        if (reasoning) {
            const cleanReasoning = reasoning
                .replace(/<think>|<reasoning>/gi, '')
                .replace(/<\/think>|<\/reasoning>/gi, '');
            
            html += `
                <div class="reasoning-block">
                    <div class="reasoning-header">
                        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M12 16v-4m0-4h.01"></path>
                        </svg>
                        <span>Reasoning</span>
                    </div>
                    <div class="reasoning-content">${escapeHtml(cleanReasoning)}</div>
                </div>
            `;
        }
        
        html += formatDemoMessage(text);
        contentDiv.innerHTML = html;
        
        if (typeof hljs !== 'undefined') {
            contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        }
        
        addDemoCopyButtons(contentDiv);
        addMessageActions(contentDiv, text);
    } else {
        messageDiv.innerHTML = `
            <div class="message system-message">
                <div class="message-content">${escapeHtml(text)}</div>
            </div>
        `;
    }
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════════
// ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════
function handleError(errorMsg, retryable = false) {
    const container = document.getElementById('messages-container');
    if (!container) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-row assistant-row';
    
    let errorHTML = `
        <div class="message error-message">
            <div class="avatar">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                </svg>
            </div>
            <div class="message-content">
                <div><strong>⚠️ Error</strong></div>
                <div>${escapeHtml(errorMsg)}</div>
    `;
    
    if (retryable && lastUserMessage) {
        errorHTML += `
            <button onclick="retryLastMessage()" class="btn btn--sm btn--primary" style="margin-top: 12px;">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                    <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z"/>
                </svg>
                Retry
            </button>
        `;
    }
    
    errorHTML += `
            </div>
        </div>
    `;
    
    messageDiv.innerHTML = errorHTML;
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;
}

window.retryLastMessage = async function() {
    if (!lastUserMessage) return;
    console.log('🔄 Retrying last message');
    await sendMessage(lastUserMessage);
};

// ═══════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

function formatDemoMessage(text) {
    if (typeof marked === 'undefined') {
        return escapeHtml(text).replace(/\n/g, '<br>');
    }
    
    marked.setOptions({
        highlight(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs ? hljs.highlightAuto(code).value : code;
        },
        breaks: true,
        gfm: true
    });
    
    return marked.parse(text);
}

function addDemoCopyButtons(container) {
    container.querySelectorAll('pre code').forEach((codeBlock) => {
        const pre = codeBlock.parentElement;
        if (pre.querySelector('.copy-code-btn')) return;
        
        const btn = document.createElement('button');
        btn.className = 'copy-code-btn';
        btn.textContent = 'Copy';
        btn.addEventListener('click', () => {
            navigator.clipboard.writeText(codeBlock.textContent).then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            });
        });
        
        pre.style.position = 'relative';
        pre.appendChild(btn);
    });
}

// ═══════════════════════════════════════════════════════════════════
// MESSAGE ACTIONS
// ═══════════════════════════════════════════════════════════════════
function addMessageActions(contentDiv, text) {
    if (contentDiv.querySelector('.message-actions')) return;
    
    const actions = document.createElement('div');
    actions.className = 'message-actions';
    actions.innerHTML = `
        <button class="msg-btn" data-action="listen" title="Read aloud">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
            Listen
        </button>
        <button class="msg-btn" data-action="copy" title="Copy response">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy
        </button>
        <button class="msg-btn" data-action="like" title="Good response">
            👍 Like
        </button>
        <button class="msg-btn" data-action="dislike" title="Bad response">
            👎 Dislike
        </button>
    `;
    
    actions.addEventListener('click', (e) => {
        const btn = e.target.closest('.msg-btn');
        if (btn) {
            handleMessageAction(btn.dataset.action, text, actions);
        }
    });
    
    contentDiv.appendChild(actions);
}

function handleMessageAction(action, text, actionsEl) {
    if (action === 'listen') {
        try {
            if (speechSynthesis.speaking) {
                speechSynthesis.cancel();
                return;
            }
            const utter = new SpeechSynthesisUtterance(text);
            utter.rate = 1.0;
            utter.pitch = 1.0;
            speechSynthesis.speak(utter);
        } catch (e) {
            alert('Text-to-speech is not supported in your browser');
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
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '✓ Copied';
                setTimeout(() => { btn.innerHTML = originalHTML; }, 1500);
            }
        }).catch(() => {
            alert('Failed to copy to clipboard');
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
// TYPING INDICATOR
// ═══════════════════════════════════════════════════════════════════
function showTypingIndicator() {
    const container = document.getElementById('messages-container');
    if (!container || document.getElementById('typing-indicator')) return;
    
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
        </div>
    `;
    
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

// ═══════════════════════════════════════════════════════════════════
// MODAL HANDLERS
// ═══════════════════════════════════════════════════════════════════
function openTermsModal() {
    const modal = document.getElementById('terms-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    document.documentElement.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
}

function closeTermsModal() {
    const modal = document.getElementById('terms-modal');
    if (!modal) return;
    modal.style.display = 'none';
    document.documentElement.style.overflow = '';
    document.body.style.overflow = '';
}

document.addEventListener('click', (e) => {
    const modal = document.getElementById('terms-modal');
    if (modal && e.target === modal) {
        closeTermsModal();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (isResponding) {
            stopAIResponse();
        }
        closeTermsModal();
    }
});
