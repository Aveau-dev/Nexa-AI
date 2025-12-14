// UI helpers + safe HTML
window.UI = (function () {
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  }

  function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const main = document.getElementById("main-content");
    if (!sidebar || !main) return;

    // mobile/tablet: show overlay style
    if (window.innerWidth <= 1024) {
      sidebar.classList.toggle("show");
      return;
    }

    // desktop: collapse
    sidebar.classList.toggle("collapsed");
    main.classList.toggle("sidebar-collapsed");
  }

  function toggleUserMenu() {
    const menu = document.getElementById("user-menu");
    if (!menu) return;
    menu.style.display = menu.style.display === "block" ? "none" : "block";
  }

  function closeUserMenu() {
    const menu = document.getElementById("user-menu");
    if (menu) menu.style.display = "none";
  }

  function closeModelSelector() {
    const selector = document.getElementById("model-selector");
    if (selector) selector.style.display = "none";
  }

  // close dropdowns on outside click
  document.addEventListener("click", (e) => {
    const userBtn = e.target.closest(".user-avatar");
    const menu = document.getElementById("user-menu");
    if (menu && menu.style.display === "block" && !userBtn && !menu.contains(e.target)) {
      closeUserMenu();
    }

    const selectorBtn = e.target.closest(".model-selector-btn");
    const selector = document.getElementById("model-selector");
    if (selector && selector.style.display === "block" && !selectorBtn && !selector.contains(e.target)) {
      closeModelSelector();
    }

    // mobile sidebar: close if clicked outside
    if (window.innerWidth <= 1024) {
      const sidebar = document.getElementById("sidebar");
      const toggleBtn = document.getElementById("sidebar-toggle");
      if (sidebar && sidebar.classList.contains("show")) {
        const clickedInsideSidebar = sidebar.contains(e.target);
        const clickedToggle = toggleBtn && toggleBtn.contains(e.target);
        if (!clickedInsideSidebar && !clickedToggle) sidebar.classList.remove("show");
      }
    }
  });

  return {
    escapeHtml,
    toggleSidebar,
    toggleUserMenu,
    closeUserMenu,
    closeModelSelector,
  };
})();
