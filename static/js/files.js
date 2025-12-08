/* static/js/files.js
   Files module:
   - Upload files (images, pdfs, csv)
   - List files (GET /api/files)
   - Preview files via /uploads/<filename>
   - Analyze file via /analyze-file?file=
*/

const Files = (function () {
  const fileListElId = "files-list";
  const filePreviewId = "file-preview";
  const fileNameId = "file-name";
  const fileAnalysisId = "file-analysis";
  const uploadInputId = "file-upload-input";

  async function init() {
    await loadFiles();
  }

  async function uploadDialog() {
    const inp = document.getElementById(uploadInputId);
    if (inp) inp.click();
  }

  async function upload(e) {
    const files = e.target.files;
    if (!files || !files.length) return;
    const file = files[0];
    const form = new FormData();
    form.append("file", file);

    try {
      UI.showViewLoader(true);
      const resp = await fetch("/upload", { method: "POST", body: form });
      const data = await resp.json();
      if (data.error) {
        UI.toast("Upload failed: " + data.error);
      } else {
        UI.toast("Upload successful");
        await loadFiles();
      }
    } catch (err) {
      console.error("Files.upload error:", err);
      UI.toast("Upload error");
    } finally {
      UI.showViewLoader(false);
    }
  }

  async function loadFiles() {
    const listEl = document.getElementById(fileListElId);
    if (!listEl) return;
    listEl.innerHTML = "<div class='loading'>Loading files…</div>";

    try {
      const resp = await fetch("/api/files");
      if (!resp.ok) {
        listEl.innerHTML = "<div class='muted'>No files</div>";
        return;
      }
      const files = await resp.json();
      if (!Array.isArray(files) || files.length === 0) {
        listEl.innerHTML = "<div class='muted'>No files uploaded yet.</div>";
        return;
      }
      listEl.innerHTML = files.map(f => {
        return `<div class="file-item" data-filename="${f.filename}" onclick="Files.select('${f.filename}')">
          <div class="file-name">${f.filename}</div>
          <div class="file-meta">${f.size ? f.size : ""} ${f.created_at ? f.created_at : ""}</div>
        </div>`;
      }).join("");
    } catch (e) {
      console.error("Files.loadFiles error:", e);
      listEl.innerHTML = "<div class='muted'>Failed to load files</div>";
    }
  }

  async function select(filename) {
    const viewer = document.getElementById(filePreviewId);
    const fnameEl = document.getElementById(fileNameId);
    const analysisEl = document.getElementById(fileAnalysisId);
    if (!viewer || !fnameEl || !analysisEl) return;

    fnameEl.textContent = filename;
    viewer.innerHTML = "Loading preview…";
    analysisEl.innerHTML = "Loading analysis…";

    const url = `/uploads/${encodeURIComponent(filename)}`;

    // preview heuristics
    if (filename.match(/\.(png|jpg|jpeg|gif|webp)$/i)) {
      viewer.innerHTML = `<img src="${url}" style="max-width:100%"/>`;
    } else if (filename.match(/\.pdf$/i)) {
      viewer.innerHTML = `<iframe src="${url}" style="width:100%;height:600px;border:none"></iframe>`;
    } else {
      // text fallback
      try {
        const r = await fetch(url);
        const txt = await r.text();
        viewer.innerHTML = `<pre style="max-height:500px;overflow:auto">${UI.escape(txt.slice(0, 3000))}</pre>`;
      } catch (e) {
        viewer.innerHTML = "Preview not available";
      }
    }

    // attempt analysis call (backend implement analyze-file)
    try {
      const resp = await fetch(`/analyze-file?file=${encodeURIComponent(filename)}`);
      if (!resp.ok) {
        analysisEl.innerHTML = "No analysis available";
        return;
      }
      const data = await resp.json();
      analysisEl.innerHTML = `<pre>${UI.escape(JSON.stringify(data, null, 2))}</pre>`;
    } catch (e) {
      console.error("Files.select analysis error:", e);
      analysisEl.innerHTML = "Analysis failed";
    }
  }

  async function analyze() {
    const fname = document.getElementById(fileNameId)?.textContent;
    if (!fname) return UI.toast("Select a file first");
    try {
      UI.showViewLoader(true);
      await select(fname); // re-run select to fetch analysis
      UI.toast("Analysis complete");
    } catch (e) {
      console.error("Files.analyze", e);
    } finally {
      UI.showViewLoader(false);
    }
  }

  function download() {
    const fname = document.getElementById(fileNameId)?.textContent;
    if (!fname) return UI.toast("Select a file first");
    const url = `/uploads/${encodeURIComponent(fname)}`;
    window.open(url, "_blank");
  }

  return {
    init,
    uploadDialog,
    upload,
    loadFiles,
    select,
    analyze,
    download
  };
})();

window.Files = Files;
