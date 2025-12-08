/* ============================================================
   SPA ROUTER — NexaAI
   Loads views dynamically from /load-view/<view>
   ============================================================ */

const Router = {
    currentView: null,
    viewContainer: null,

    init() {
        this.viewContainer = document.getElementById("view-container");

        // Load initial view
        const initialView = window.location.pathname.replace("/dashboard/", "") || "chat";
        this.load(initialView, false);

        // Handle browser back/forward buttons
        window.onpopstate = (event) => {
            const view = event.state?.view || "chat";
            this.load(view, false);
        };

        console.log("Router initialized");
    },

    /* ============================================================
       LOAD VIEW
       ------------------------------------------------------------ */
    async load(view, pushState = true) {
        try {
            UI.showViewLoader(true);

            const response = await fetch(`/load-view/${view}`);
            const html = await response.text();

            this.viewContainer.style.opacity = "0";

            setTimeout(() => {
                // Replace content
                this.viewContainer.innerHTML = html;
                this.viewContainer.style.opacity = "1";
            }, 150);

            this.currentView = view;

            // Set URL
            if (pushState) {
                history.pushState({ view }, "", `/dashboard/${view}`);
            }

            // Activate sidebar item
            this.highlightActive(view);

            // Re-run scripts inside loaded view if needed
            this.executeInlineScripts();

        } catch (err) {
            console.error("View load error:", err);
            this.showError("Failed to load view.");
        } finally {
            UI.showViewLoader(false);
        }
    },

    /* ============================================================
       HIGHLIGHT ACTIVE SIDEBAR ITEM
       ------------------------------------------------------------ */
    highlightActive(view) {
        const items = document.querySelectorAll(".sidebar-item");
        items.forEach(i => i.classList.remove("active"));

        const activeItem = document.querySelector(`[data-view="${view}"]`);
        if (activeItem) activeItem.classList.add("active");
    },

    /* ============================================================
       EXECUTE INLINE SCRIPTS IN LOADED HTML
       ------------------------------------------------------------ */
    executeInlineScripts() {
        const scripts = this.viewContainer.querySelectorAll("script");
        scripts.forEach(oldScript => {
            const newScript = document.createElement("script");
            if (oldScript.src) newScript.src = oldScript.src;
            else newScript.innerHTML = oldScript.innerHTML;
            document.body.appendChild(newScript);
            oldScript.remove();
        });
    },

    /* ============================================================
       ERROR VIEW
       ------------------------------------------------------------ */
    showError(message) {
        this.viewContainer.innerHTML = `
            <div class="error-screen">
                <h2>Error</h2>
                <p>${message}</p>
                <button class="btn-primary" onclick="Router.load('chat')">
                    Go Back to Chat
                </button>
            </div>
        `;
    }
};

window.Router = Router;
