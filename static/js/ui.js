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

    if (window.innerWidth <= 1024) {
      sidebar.classList.toggle("show");
    } else {
      sidebar.classList.toggle("collapsed");
      main.classList.toggle("sidebar-collapsed");
    }
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

  // Close dropdowns / sidebar when clicking outside
  document.addEventListener("click", (event) => {
    const menu = document.getElementById("user-menu");
    const userBtn = event.target.closest(".user-avatar");

    if (menu && menu.style.display === "block" && !userBtn && !menu.contains(event.target)) {
      closeUserMenu();
    }

    const selector = document.getElementById("model-selector");
    const modelBtn = event.target.closest(".model-selector-btn");
    if (selector && selector.style.display === "block" && !modelBtn && !selector.contains(event.target)) {
      closeModelSelector();
    }

    if (window.innerWidth <= 1024) {
      const sidebar = document.getElementById("sidebar");
      const sidebarToggle = document.getElementById("sidebar-toggle");
      if (sidebar && sidebar.classList.contains("show")) {
        const clickedInsideSidebar = sidebar.contains(event.target);
        const clickedToggle = sidebarToggle && sidebarToggle.contains(event.target);
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
