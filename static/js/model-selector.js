window.ModelSelector = (function () {
  function toggle() {
    const selector = document.getElementById("model-selector");
    if (!selector) return;
    selector.style.display = selector.style.display === "block" ? "none" : "block";
  }

  async function select(modelKey, modelName) {
    // update UI
    const selectedName = document.getElementById("selected-model-name");
    const modelInfo = document.getElementById("model-info");
    if (selectedName) selectedName.textContent = modelName || modelKey;
    if (modelInfo) modelInfo.textContent = modelName || modelKey;

    // persist server-side (your app has /set-model) [file:349]
    try {
      await fetch("/set-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelKey }),
      });
    } catch (_) {}

    // update chat state
    Chat.setModel(modelKey, modelName);

    toggle();
  }

  return { toggle, select };
})();
