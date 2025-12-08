/* static/js/canvas.js
   Canvas / Code Workspace module
   - Lazy-loads Monaco editor from CDN if not present
   - Runs code via backend endpoint: POST /execute-code (expects JSON { code })
   - Renders markdown preview using marked
*/

const Canvas = (function () {
  let editor = null;
  let monacoLoaded = false;
  const monacoUrl = "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.41.0/min/vs/loader.min.js";
  const editorElementId = "monaco-editor";
  const markdownElId = "canvas-markdown";
  const outputElId = "canvas-output";

  async function loadMonaco() {
    if (monacoLoaded) return;
    if (window.require && window.monaco) {
      monacoLoaded = true;
      return;
    }

    // Inject loader script
    await new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = monacoUrl;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("Failed to load Monaco loader"));
      document.head.appendChild(s);
    });

    // Configure require and load editor
    return new Promise((resolve, reject) => {
      if (!window.require) return reject(new Error("Monaco loader not available"));
      window.require.config({ paths: { vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.41.0/min/vs" }});
      window.require(["vs/editor/editor.main"], () => {
        monacoLoaded = true;
        resolve();
      }, (err) => reject(err));
    });
  }

  async function init() {
    try {
      const el = document.getElementById(editorElementId);
      if (!el) return;

      // load monaco then create editor
      await loadMonaco();
      editor = monaco.editor.create(el, {
        value: "# Write Python code or Markdown\n",
        language: "python",
        theme: "vs-dark",
        automaticLayout: true,
        minimap: { enabled: false },
        fontSize: 13,
      });

      // preview markdown on change (throttled)
      let timeout = null;
      editor.onDidChangeModelContent(() => {
        clearTimeout(timeout);
        timeout = setTimeout(renderMarkdownPreview, 500);
      });

      renderMarkdownPreview();
      console.log("Canvas: Monaco ready");
    } catch (e) {
      console.error("Canvas.init error:", e);
      const container = document.getElementById(editorElementId);
      if (container) container.innerHTML = "<pre style='padding:1rem'>Editor failed to load. Check network or include Monaco manually.</pre>";
    }
  }

  function getCode() {
    if (editor) return editor.getValue();
    // fallback to a textarea if needed
    const ta = document.querySelector("#artifact-code") || document.querySelector("textarea#artifact-code");
    return ta ? ta.value : "";
  }

  async function run() {
    const code = getCode();
    if (!code) {
      UI.toast("No code to run");
      return;
    }

    const out = document.getElementById(outputElId);
    if (out) out.textContent = "Running…";

    try {
      const resp = await fetch("/execute-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code })
      });
      const data = await resp.json();
      if (data.error) {
        if (out) out.textContent = "Error: " + data.error;
      } else {
        if (out) out.textContent = data.output || JSON.stringify(data, null, 2);
      }
      UI.toast("Execution finished");
    } catch (e) {
      console.error("Canvas.run error:", e);
      if (out) out.textContent = "Execution failed: " + e.message;
      UI.toast("Execution failed");
    }
  }

  function renderMarkdownPreview() {
    const mdEl = document.getElementById(markdownElId);
    if (!mdEl) return;
    const code = getCode();
    // Basic heuristic: if starts with # or contains markdown, render as markdown
    const isLikelyMarkdown = code.trim().startsWith("#") || code.includes("```") || code.includes("- ");
    if (!isLikelyMarkdown) {
      mdEl.innerHTML = "<div style='opacity:.7'>No markdown detected</div>";
      return;
    }
    try {
      const rendered = marked.parse(code);
      mdEl.innerHTML = rendered;
      // highlight codeblocks
      mdEl.querySelectorAll("pre code").forEach((block) => {
        if (window.hljs) hljs.highlightElement(block);
      });
    } catch (e) {
      mdEl.innerHTML = "<pre>Markdown render failed</pre>";
    }
  }

  function exportArtifact() {
    const code = getCode();
    const blob = new Blob([code], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "artifact.py";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    UI.toast("Exported artifact");
  }

  function clear() {
    if (editor) editor.setValue("");
    const out = document.getElementById(outputElId);
    if (out) out.textContent = "Run code to see output...";
    UI.toast("Canvas cleared");
  }

  return {
    init,
    run,
    exportArtifact,
    clear
  };
})();

window.Canvas = Canvas;
