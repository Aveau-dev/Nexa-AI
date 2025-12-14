window.Files = (function () {
  let selectedUploadPath = null; // server filepath returned by /upload
  let selectedUploadName = null;

  function onViewLoaded() {
    const input = document.getElementById("file-input");
    if (input) {
      input.onchange = async (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) await upload(file);
      };
    }
    renderStatus();
  }

  function renderStatus(text) {
    const el = document.getElementById("files-status");
    if (!el) return;
    if (text) {
      el.textContent = text;
      return;
    }
    if (selectedUploadPath) {
      el.textContent = `Selected: ${selectedUploadName} (will be attached to next chat message)`;
    } else {
      el.textContent = "Select a file to upload. After upload, it can be attached to your next chat message.";
    }
  }

  function openPicker() {
    const input = document.getElementById("file-input");
    if (!input) return alert("File input not found in Files view.");
    input.click();
  }

  async function upload(file) {
    renderStatus("Uploading...");

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch("/upload", { method: "POST", body: fd });
      const data = await res.json();

      if (!res.ok || data.error) {
        renderStatus(data.error || "Upload failed.");
        return;
      }

      selectedUploadPath = data.filepath || data.path || null;
      selectedUploadName = data.filename || file.name;

      renderStatus();
    } catch (e) {
      renderStatus("Upload failed. Please try again.");
    }
  }

  function refresh() {
    renderStatus();
    alert("Files list API not implemented in this template yet.");
  }

  function getSelectedUploadPath() {
    return selectedUploadPath;
  }

  function clearSelectedUpload() {
    selectedUploadPath = null;
    selectedUploadName = null;
    renderStatus();
  }

  return {
    onViewLoaded,
    openPicker,
    upload,
    refresh,
    getSelectedUploadPath,
    clearSelectedUpload,
  };
})();
