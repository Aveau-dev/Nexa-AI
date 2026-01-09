'use strict';

// ═══════════════════════════════════════════════════════════════════
// STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════
let isResponding = false;
let abortController = null;
let lastUserMessage = '';
let currentFile = null;
let currentFileType = null;
let currentFileBase64 = null;
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
    
    console.log('🚀 NexaAI Demo - Qwen 2.5 VL with Vision + Image Gen');
});

// ═══════════════════════════════════════════════════════════════════
// FILE HANDLING
// ═══════════════════════════════════════════════════════════════════
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File too large. Maximum size is 10MB.');
        return;
    }
    
    currentFile = file;
    currentFileType = file.type;
    
    // Convert to base64
    const reader = new FileReader();
    reader.onload = (e) => {
        currentFileBase64 = e.target.result.split(',')[1];
        showFilePreview(file.name);
    };
    reader.readAsDataURL(file);
}

function showFilePreview(filename) {
    const uploadArea = document.getElementById('file-upload-area');
    const preview = document.getElementById('file-preview');
    
    if (!uploadArea || !preview) return;
    
    const fileIcon = currentFileType.startsWith('image/') ? '🖼️' : '📄';
    const displayName = filename.length > 30 ? filename.substring(0, 27) + '...' : filename;
    
    preview.innerHTML = `
        <div class="file-preview-item">
            <span class="file-preview-icon">${fileIcon}</span>
            <span class="file-preview-name">${displayName}</span>
            <button class="file-remove-btn" onclick="clearFile()">✕</button>
        </div>
    `;
    
    uploadArea.classList.add('active');
}

window.clearFile = function() {
    currentFile = null;
    currentFileType = null;
    currentFileBase64 = null;
    
    const uploadArea = document.getElementById('file-upload-area');
    const fileInput = document.getElementById('file-input');
    
    if (uploadArea) uploadArea.classList.remove('active');
    if (fileInput) fileInput.value = '';
};

// ═══════════════════════════════════════════════════════════════════
// IMAGE GENERATION MODE
// ═══════════════════════════════════════════════════════════════════
function toggleImageGenMode() {
    imageGenMode = !imageGenMode;
    const btn = document.getElementById('image-gen-btn');
    const input = document.getElementById('chat-input');
    
    if (!btn || !input) return;
    
    if (imageGenMode) {
        btn.style.background = 'var(--color-primary)';
        btn.style.color = 'var(--color-btn-primary-text)';
        input.placeholder = '✨ Describe the image you want to generate...';
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
    addSystemMessage('⏸️ Response stopped by user');
    setInputState(false);
    isResponding = false;
    abortController = null;
}

// ═══════════════════════════════════════════════════════════════════
// SEND MESSAGE (ENHANCED)
// ═══════════════════════════════════════════════════════════════════
async function sendMessage(retryMessage = null) {
    const input = document.getElementById('chat-input');
    const message = retryMessage || (input ? input.value.trim() : '');
    
    if (!message && !currentFileBase64) {
        return;
    }
    
    lastUserMessage = message;
    
    // Hide welcome section
    const welcome = document.getElementById('welcome-section');
    if (welcome) {
        welcome.style.display = 'none';
    }
    
    // Check for image generation keywords (automatic detection)
    const imageKeywords = ['draw', 'generate image', 'create image', 'make image', 'paint', 'show me', 'illustrate'];
    const autoDetectImage = imageKeywords.some(kw => message.toLowerCase().includes(kw));
    
    // Handle Image Generation Mode or auto-detection
    if ((imageGenMode || autoDetectImage) && message && !currentFileBase64) {
        addMessageUI(message, 'user');
        if (input && !retryMessage) {
            input.value = '';
            input.style.height = 'auto';
        }
        if (imageGenMode) {
            imageGenMode = false;
            toggleImageGenMode();
        }
        await generateImage(message);
        return;
    }
    
    // Display user message with file indicator
    let userMessageDisplay = message;
    if (currentFileBase64 && currentFileType.startsWith('image/')) {
        userMessageDisplay += '\n\n🖼️ *Image attached for analysis*';
    } else if (currentFileBase64) {
        userMessageDisplay += '\n\n📄 *File attached for analysis*';
    }
    addMessageUI(userMessageDisplay, 'user');
    
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
            context: getContextMessages()
        };
        
        // Add image data if present
        if (currentFileBase64 && currentFileType.startsWith('image/')) {
            payload.image = currentFileBase64;
        }
        
        const res = await fetch('/demo-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: abortController.signal
        });
        
        // Clear file after sending
        if (currentFileBase64) {
            window.clearFile();
        }
        
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
// GET CONTEXT MESSAGES
// ═══════════════════════════════════════════════════════════════════
function getContextMessages() {
    const container = document.getElementById('messages-container');
    if (!container) return [];
    
    const messages = [];
    const messageRows = container.querySelectorAll('.message-row:not(.typing-indicator):not(.system-row)');
    
    // Get last 6 messages for context
    const recentMessages = Array.from(messageRows).slice(-6);
    
    recentMessages.forEach(row => {
        const isUser = row.classList.contains('user-row');
        const content = row.querySelector('.message-content');
        if (content) {
            let text = content.textContent || content.innerText || '';
            // Remove file indicators
            text = text.replace(/🖼️.*Image attached.*\*/gi, '').trim();
            text = text.replace(/📄.*File attached.*\*/gi, '').trim();
            
            messages.push({
                role: isUser ? 'user' : 'assistant',
                content: text
            });
        }
    });
    
    return messages;
}

// ═══════════════════════════════════════════════════════════════════
// IMAGE GENERATION
// ═══════════════════════════════════════════════════════════════════
async function generateImage(prompt) {
    showTypingIndicator();
    setInputState(true);
    
    try {
        // Clean prompt for image generation
        let cleanPrompt = prompt.toLowerCase();
        const prefixes = ['draw', 'generate image', 'create image', 'make image', 'paint', 'show me', 'illustrate'];
        prefixes.forEach(prefix => {
            cleanPrompt = cleanPrompt.replace(prefix, '').trim();
        });
        cleanPrompt = cleanPrompt || prompt;
        
        // Using Pollinations AI (free and reliable)
        const encoded = encodeURIComponent(cleanPrompt);
        const seed = Math.floor(Math.random() * 100000);
        const imageUrl = `https://image.pollinations.ai/prompt/${encoded}?width=1024&height=1024&nologo=true&seed=${seed}`;
        
        // Wait a moment for image to generate
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        removeTypingIndicator();
        
        // Display generated image with actions
        const imageHTML = `
<p style="margin-bottom: 12px;">✨ I've generated an image based on your description:</p>
<div class="generated-image-container">
    <img src="${imageUrl}" 
         alt="Generated image" 
         class="message-image" 
         onclick="window.open('${imageUrl}', '_blank')"
         onerror="this.parentElement.innerHTML='<p>⚠️ Image failed to load. Please try again.</p>'">
</div>
<div class="image-actions">
    <button class="btn-small" onclick="downloadImage('${imageUrl}')">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
        </svg>
        Download
    </button>
    <button class="btn-small" onclick="window.open('${imageUrl}', '_blank')">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6v6M10 14L21 3"/>
        </svg>
        Open Full Size
    </button>
    <button class="btn-small" onclick="regenerateImage('${cleanPrompt}')">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0118.8-4.3M22 12.5a10 10 0 01-18.8 4.2"/>
        </svg>
        Regenerate
    </button>
</div>
<p style="font-size: 13px; color: var(--color-text-secondary); margin-top: 8px;">
    💡 Click image to view full size · Powered by Pollinations AI
</p>`;
        
        addMessageUI(imageHTML, 'assistant', null, true);
        setInputState(false);
    } catch (error) {
        console.error('Image generation error:', error);
        removeTypingIndicator();
        handleError('Failed to generate image. Please try again.', true);
        setInputState(false);
    }
}

window.downloadImage = function(url) {
    const a = document.createElement('a');
    a.href = url;
    a.download = `nexa-ai-image-${Date.now()}.png`;
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
};

window.regenerateImage = function(prompt) {
    const input = document.getElementById('chat-input');
    if (input) {
        input.value = `generate image of ${prompt}`;
    }
    sendMessage();
};

// ═══════════════════════════════════════════════════════════════════
// SSE STREAMING (ENHANCED)
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
                            }
                            
                            // 👇 FIX: Trim whitespace before updating
                            updateStreamingMessage(messageElement, fullResponse.trimStart(), reasoning);
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
            addMessageUI(fullResponse.trim(), 'assistant', reasoning);
        }
        setInputState(false);
        abortController = null;
    } catch (error) {
        if (error.name === 'AbortError') return;
        console.error('Stream error:', error);
        removeTypingIndicator();
        setInputState(false);
        abortController = null;
    }
}

// ═══════════════════════════════════════════════════════════════════
// UI FUNCTIONS (ENHANCED WITH FIX)
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
            <div class="message-content markdown-content"></div>
        </div>
    `;
    
    container.appendChild(messageDiv);
    scrollToBottom();
    
    return messageDiv.querySelector('.markdown-content');
}

function updateStreamingMessage(element, text, reasoning = '') {
    if (!element) return;
    
    // 👇 FIX: Trim text to remove leading/trailing whitespace
    const trimmedText = text.trim();
    let html = '';
    
    // Add reasoning block if present
    if (reasoning) {
        const cleanReasoning = reasoning
            .replace(/<think>|<reasoning>/gi, '')
            .replace(/<\/think>|<\/reasoning>/gi, '')
            .trim();
        
        if (cleanReasoning) {
            html += `
<details class="reasoning-block" open>
    <summary class="reasoning-header">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
        <span>💭 Reasoning Process</span>
    </summary>
    <div class="reasoning-content">${formatMarkdown(cleanReasoning)}</div>
</details>`;
        }
    }
    
    html += formatMarkdown(trimmedText);
    element.innerHTML = html;
    
    // Syntax highlighting
    if (typeof hljs !== 'undefined') {
        element.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    }
    
    addCopyButtons(element);
    scrollToBottom();
}

function addMessageUI(text, role, reasoning = '', isHTML = false) {
    const container = document.getElementById('messages-container');
    if (!container) return;
    
    // 👇 FIX: Trim text before processing
    const trimmedText = text.trim();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message-row ${role}-row`;
    
    if (role === 'user') {
        messageDiv.innerHTML = `
            <div class="message user-message">
                <div class="message-content">${escapeHtml(trimmedText).replace(/\n/g, '<br>')}</div>
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
                <div class="message-content markdown-content"></div>
            </div>
        `;
        
        const contentDiv = messageDiv.querySelector('.markdown-content');
        let html = '';
        
        // Add reasoning block if present
        if (reasoning) {
            const cleanReasoning = reasoning
                .replace(/<think>|<reasoning>/gi, '')
                .replace(/<\/think>|<\/reasoning>/gi, '')
                .trim();
            
            if (cleanReasoning) {
                html += `
<details class="reasoning-block">
    <summary class="reasoning-header">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
        </svg>
        <span>💭 Reasoning Process</span>
    </summary>
    <div class="reasoning-content">${isHTML ? cleanReasoning : formatMarkdown(cleanReasoning)}</div>
</details>`;
            }
        }
        
        html += isHTML ? trimmedText : formatMarkdown(trimmedText);
        contentDiv.innerHTML = html;
        
        // Syntax highlighting
        if (typeof hljs !== 'undefined') {
            contentDiv.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }
        
        addCopyButtons(contentDiv);
        addMessageActions(contentDiv, trimmedText);
    }
    
    container.appendChild(messageDiv);
    scrollToBottom();
}

function addSystemMessage(text) {
    const container = document.getElementById('messages-container');
    if (!container) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message-row system-row';
    messageDiv.innerHTML = `
        <div class="message system-message">
            <div class="message-content">${escapeHtml(text)}</div>
        </div>
    `;
    
    container.appendChild(messageDiv);
    scrollToBottom();
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
                <div style="font-weight: 600; margin-bottom: 4px;">⚠️ Error</div>
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
    
    errorHTML += '</div></div>';
    messageDiv.innerHTML = errorHTML;
    container.appendChild(messageDiv);
    scrollToBottom();
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

function formatMarkdown(text) {
    if (typeof marked === 'undefined') {
        return escapeHtml(text).replace(/\n/g, '<br>');
    }
    
    // 👇 FIX: Trim text before rendering
    const trimmed = text.trim();
    
    marked.setOptions({
        highlight(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (e) {
                    console.warn('Highlight error:', e);
                }
            }
            return typeof hljs !== 'undefined' ? hljs.highlightAuto(code).value : escapeHtml(code);
        },
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });
    
    try {
        const rendered = marked.parse(trimmed);
        // 👇 FIX: Trim rendered HTML to remove extra whitespace
        return rendered.trim();
    } catch (e) {
        console.error('Markdown parse error:', e);
        return escapeHtml(trimmed).replace(/\n/g, '<br>');
    }
}

function addCopyButtons(container) {
    container.querySelectorAll('pre code').forEach((codeBlock) => {
        const pre = codeBlock.parentElement;
        if (pre.querySelector('.copy-code-btn')) return;
        
        const btn = document.createElement('button');
        btn.className = 'copy-code-btn';
        btn.textContent = 'Copy';
        btn.addEventListener('click', () => {
            navigator.clipboard.writeText(codeBlock.textContent).then(() => {
                btn.textContent = '✓ Copied';
                setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
            }).catch(() => {
                btn.textContent = '✗ Failed';
                setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
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
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
        </button>
        <button class="msg-btn" data-action="copy" title="Copy response">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        </button>
        <button class="msg-btn" data-action="like" title="Good response">
            👍
        </button>
        <button class="msg-btn" data-action="dislike" title="Bad response">
            👎
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
    const plainText = text.replace(/<[^>]*>/g, '').replace(/\n\n+/g, '\n');
    
    if (action === 'listen') {
        try {
            if (speechSynthesis.speaking) {
                speechSynthesis.cancel();
                return;
            }
            const utter = new SpeechSynthesisUtterance(plainText);
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
        navigator.clipboard.writeText(plainText).then(() => {
            const btn = actionsEl.querySelector('[data-action="copy"]');
            if (btn) {
                const originalHTML = btn.innerHTML;
                btn.innerHTML = '✓';
                setTimeout(() => { btn.innerHTML = originalHTML; }, 2000);
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
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    const container = document.getElementById('messages-container');
    if (container) {
        requestAnimationFrame(() => {
            container.scrollTop = container.scrollHeight;
        });
    }
}

// ═══════════════════════════════════════════════════════════════════
// MODAL HANDLERS
// ═══════════════════════════════════════════════════════════════════
window.openTermsModal = function() {
    const modal = document.getElementById('terms-modal');
    if (!modal) return;
    modal.style.display = 'flex';
    document.documentElement.style.overflow = 'hidden';
    document.body.style.overflow = 'hidden';
};

window.closeTermsModal = function() {
    const modal = document.getElementById('terms-modal');
    if (!modal) return;
    modal.style.display = 'none';
    document.documentElement.style.overflow = '';
    document.body.style.overflow = '';
};

// Close modal on outside click
document.addEventListener('click', (e) => {
    const modal = document.getElementById('terms-modal');
    if (modal && e.target === modal) {
        window.closeTermsModal();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (isResponding) {
            stopAIResponse();
        }
        window.closeTermsModal();
    }
});

console.log('✅ NexaAI Demo - Qwen 2.5 VL Enhanced - Loaded Successfully');
