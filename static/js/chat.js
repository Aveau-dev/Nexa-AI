/* ============================================================
   NexaAI Chat Module
   ChatGPT-style streaming, markdown rendering, image attachments
   ============================================================ */

const Chat = {
    messagesBox: null,
    input: null,
    attachedImage: null,

    init() {
        this.messagesBox = document.getElementById("chat-messages");
        this.input = document.getElementById("global-input");

        console.log("Chat initialized");
    },

    /* ============================================================
       Send message
       ============================================================ */
    async send() {
        const text = this.input.value.trim();
        if (!text && !this.attachedImage) return;

        this.renderUserMessage(text, this.attachedImage);

        UI.clearInput();

        // Build payload
        const payload = new FormData();
        payload.append("message", text);
        payload.append("model", ModelSelector.selectedModel);

        if (this.attachedImage) {
            payload.append("image", this.attachedImage);
            this.attachedImage = null;
        }

        // Start assistant placeholder
        const assistantEl = this.renderAssistantPlaceholder();

        // Request to server
        const response = await fetch("/api/chat/send", {
            method: "POST",
            body: payload
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";

        // Live streaming
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            assistantEl.innerHTML = marked.parse(buffer);
            UI.scrollToBottom();
        }
    },

    /* ============================================================
       Render User Message
       ============================================================ */
    renderUserMessage(text, image) {
        const wrapper = document.createElement("div");
        wrapper.className = "message-row user-row";

        wrapper.innerHTML = `
