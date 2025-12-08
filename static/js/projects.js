/* static/js/projects.js
   Projects lightweight manager
   - list projects, create, open detail
   - upload reference files to a project
*/

const Projects = (function () {
  const listElId = "projects-list";
  const detailId = "project-detail";
  const projectFilesId = "project-files";
  const projectChatsId = "project-chats";

  async function init() {
    await loadProjects();
  }

  async function loadProjects() {
    const el = document.getElementById(listElId);
    if (!el) return;
    el.innerHTML = "<div class='loading'>Loading projects…</div>";
    try {
      const resp = await fetch("/api/projects");
      if (!resp.ok) {
        el.innerHTML = "<div class='muted'>No projects found</div>";
        return;
      }
      const projects = await resp.json();
      if (!projects.length) {
        el.innerHTML = "<div class='muted'>No projects yet</div>";
        return;
      }
      el.innerHTML = projects.map(p => `
        <div class="project-card" data-id="${p.id}" onclick="Projects.open(${p.id})">
          <h4>${p.title}</h4>
          <div class="muted">${p.description || ""}</div>
        </div>
      `).join("");
    } catch (e) {
      console.error("Projects.load error:", e);
      el.innerHTML = "<div class='muted'>Failed to load</div>";
    }
  }

  async function create() {
    const title = prompt("Project name:");
    if (!title) return;
    try {
      const resp = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      const data = await resp.json();
      if (data.error) {
        UI.toast("Create failed");
      } else {
        UI.toast("Project created");
        await loadProjects();
      }
    } catch (e) {
      console.error("Projects.create error", e);
      UI.toast("Create error");
    }
  }

  async function open(projectId) {
    const detail = document.getElementById(detailId);
    if (!detail) return;
    detail.classList.remove("hidden");
    try {
      const resp = await fetch(`/api/projects/${projectId}`);
      if (!resp.ok) {
        detail.innerHTML = "<div class='muted'>Failed to load project</div>";
        return;
      }
      const project = await resp.json();
      document.getElementById("project-title").textContent = project.title;
      // render files & chats
      const filesEl = document.getElementById(projectFilesId);
      filesEl.innerHTML = (project.files || []).map(f => `<div>${f.filename}</div>`).join("") || "<div class='muted'>No files</div>";

      const chatsEl = document.getElementById(projectChatsId);
      chatsEl.innerHTML = (project.chats || []).map(c => `<div>${c.title}</div>`).join("") || "<div class='muted'>No chats</div>";
    } catch (e) {
      console.error("Projects.open error:", e);
    }
  }

  function closeDetail() {
    const detail = document.getElementById(detailId);
    if (!detail) return;
    detail.classList.add("hidden");
  }

  function triggerFileUpload() {
    document.getElementById("project-file-input")?.click();
  }

  async function uploadFile(ev) {
    const files = ev.target.files;
    if (!files || !files.length) return;
    // You need to supply which project to upload to; for now assume last opened project title element contains id (extend as needed)
    const projectTitle = document.getElementById("project-title").textContent;
    if (!projectTitle) return UI.toast("Open a project first");
    // Implement the correct project id mapping in the real app
    UI.toast("Uploading file to project (backend endpoint required)");
  }

  return {
    init: init,
    loadProjects,
    create,
    open,
    closeDetail,
    triggerFileUpload,
    uploadFile
  };
})();

window.Projects = Projects;
