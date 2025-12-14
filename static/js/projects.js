window.Projects = (function () {
  const projects = [];

  function onViewLoaded() {
    render();
  }

  function openCreate() {
    const el = document.getElementById("projects-create");
    if (el) el.style.display = "block";
  }

  function closeCreate() {
    const el = document.getElementById("projects-create");
    if (el) el.style.display = "none";
  }

  function create() {
    const name = (document.getElementById("project-name")?.value || "").trim();
    const desc = (document.getElementById("project-desc")?.value || "").trim();
    if (!name) return alert("Project name required.");

    projects.unshift({ id: Date.now(), name, desc });
    closeCreate();

    const n = document.getElementById("project-name");
    const d = document.getElementById("project-desc");
    if (n) n.value = "";
    if (d) d.value = "";

    render();
  }

  function remove(id) {
    const idx = projects.findIndex((p) => p.id === id);
    if (idx >= 0) projects.splice(idx, 1);
    render();
  }

  function render() {
    const list = document.getElementById("projects-list");
    if (!list) return;

    if (projects.length === 0) {
      list.innerHTML = `
        <div class="message assistant-message" style="width:100%; max-width:48rem;">
          <div class="avatar">N</div>
          <div class="message-content">
            <div class="model-badge">Projects</div>
            <div class="message-text markdown-content">No projects yet.</div>
          </div>
        </div>
      `;
      return;
    }

    list.innerHTML = projects
      .map(
        (p) => `
        <div class="message assistant-message" style="width:100%; max-width:48rem; margin-bottom:0.75rem;">
          <div class="avatar">N</div>
          <div class="message-content">
            <div class="model-badge">${UI.escapeHtml(p.name)}</div>
            <div class="message-text markdown-content">
              <div>${UI.escapeHtml(p.desc || "No description")}</div>
              <div style="margin-top:0.75rem; display:flex; gap:0.5rem;">
                <button class="nav-button secondary" onclick="Projects.remove(${p.id})">Delete</button>
              </div>
            </div>
          </div>
        </div>
      `
      )
      .join("");
  }

  function refresh() {
    render();
  }

  return { onViewLoaded, openCreate, closeCreate, create, remove, refresh };
})();
