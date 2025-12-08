/* ============================================================
   NexaAI UI Module
   Controls global input, loaders, scrolling, toasts
   ============================================================ */

const UI = {
    inputWrapper: null,
    input: null,
    sendBtn: null,
    viewLoader: null,

    init() {
        this.inputWrapper = document.getElementById("global-input-wrapper");
        this.input = document.getElementById("global-input");
        this.sendBtn = document.getElementById("global-input-send");
        this.viewLoader = document.getElementById("view-loader");

        console.log("UI initialized");
    },

    /* ============================================================
       Enable / Disable global input
       (Chat uses input, other views hide it)
       ============================================================ */
    enableGlobalInput(state) {
        if (!this.inputWrapper) return;
        this.inputWrapper.style.display = state ? "flex" : "none";
    },

    /* ============================================================
       Loader for router view changes
       ============================================================ */
    showViewLoader(state) {
        if (!this.viewLoader) return;
        this.viewLoader.style.display = state ? "block" : "none";
    },

    /* ============================================================
       Input utilities
       ============================================================ */
    clearInput() {
        if (this.input) this.input.value = "";
    },

    setInput(text) {
        this.input.value = text;
        this.input.focus();
    },

    /* ============================================================
       Scrolling
       ============================================================ */
    scrollToBottom() {
        setTimeout(() => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
        }, 100);
    },

    /* ============================================================
       Escape HTML (safe text)
       ============================================================ */
    escape(text) {
        return text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
    },

    /* ============================================================
       Toast Notifications
       ============================================================ */
    toast(msg) {
        const el = document.createElement("div");
        el.className = "toast";
        el.innerText = msg;

        document.body.appendChild(el);

        setTimeout(() => el.classList.add("show"), 10);
        setTimeout(() => el.classList.remove("show"), 2500);
        setTimeout(() => el.remove(), 3000);
    }
};

window.UI = UI;
