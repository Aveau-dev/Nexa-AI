/* ============================================================
   Model Selector
   Handles model switching, UI badge updates
   ============================================================ */

const ModelSelector = {
    selectedModel: "gpt-5.1-mini",
    displayNameElement: null,

    init() {
        this.displayNameElement = document.getElementById("selected-model-name");
        this.updateDisplay();

        console.log("ModelSelector initialized");
    },

    /* ============================================================
       Called when user selects a model
       ============================================================ */
    select(modelId, displayName) {
        this.selectedModel = modelId;
        this.displayNameElement.textContent = displayName;

        UI.toast(`Model switched to ${displayName}`);

        // Close selector
        document.getElementById("model-selector").style.display = "none";
    },

    updateDisplay() {
        if (this.displayNameElement) {
            this.displayNameElement.textContent = "GPT-5.1 mini";
        }
    },

    toggle() {
        const el = document.getElementById("model-selector");
        el.style.display = el.style.display === "flex" ? "none" : "flex";
    }
};

window.ModelSelector = ModelSelector;
