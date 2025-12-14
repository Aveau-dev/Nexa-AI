window.Files = (function () {
  let selectedUploadPath = null;
  let selectedUploadName = null;

  function onViewLoaded() {
    const input = document.getElementById("file-input");
    if (input) {
      input.onchange = async (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) await upload(f);
      };
    }
    renderStatus();
  }

  function openPicker() {
    const input = document.getElementById("file-input");
    if (!input) return alert("Open Files view first.");
    input.click();
  }

  function renderStatus(msg) {
    const el = document.getElementById("files-status");
    if (!el) return;

    if (msg) {
      el.textContent = msg;
      return;
    }

    if (selectedUploadPath) {
      el.textContent = `Selected: ${selectedUploadName} (will attach to next chat message)`;
    } else {
      el.textContent = "No file selected. Upload a file to attach it to your next message.";
    }
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

      selectedUploadPath = data.filepath || null; // app.py returns filepath [file:349]
      selectedUploadName = data.filename || file.name;
      renderStatus();
    } catch (e) {
      renderStatus("Upload failed. Please try again.");
    }
  }

  function refresh() {
    renderStatus();
    alert("Listing files is not implemented (needs an API).");
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
