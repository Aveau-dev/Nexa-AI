/* global window, document */
(function () {
  const UI = {
    init() {
      // cache nothing mandatory; just ensure required elements exist
      this.sidebar = document.getElementById('sidebar');
      this.userMenu = document.getElementById('user-menu');
      this.modelDropdown = document.getElementById('model-selector-dropdown');
    },

    toggleSidebar() {
      const sidebar = document.getElementById('sidebar');
      const main = document.getElementById('main-container');
      if (!sidebar || !main) return;

      // mobile vs desktop
      if (window.innerWidth <= 1024) sidebar.classList.toggle('show');
      else {
        sidebar.classList.toggle('collapsed');
        main.classList.toggle('sidebar-collapsed');
      }
    },

    toggleUserMenu(force) {
      const menu = document.getElementById('user-menu');
      if (!menu) return;

      const shouldShow = (typeof force === 'boolean')
        ? force
        : (menu.style.display !== 'block');

      menu.style.display = shouldShow ? 'block' : 'none';
    },

    autoResize(textarea) {
      if (!textarea) return;
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    },

    handleGlobalOutsideClick(ev) {
      // close user menu
      const menu = document.getElementById('user-menu');
      if (menu && !ev.target.closest('.user-avatar') && !ev.target.closest('#user-menu')) {
        menu.style.display = 'none';
      }

      // close model selector
      const dd = document.getElementById('model-selector-dropdown');
      if (dd && dd.classList.contains('show')
          && !ev.target.closest('#model-selector-btn')
          && !ev.target.closest('#model-selector-dropdown')) {
        if (window.Models) window.Models.toggleSelector(false);
      }

      // close sidebar on mobile if clicking outside
      const sidebar = document.getElementById('sidebar');
      if (sidebar && window.innerWidth <= 1024 && sidebar.classList.contains('show')) {
        if (!ev.target.closest('#sidebar') && !ev.target.closest('[onclick*="toggleSidebar"]')) {
          sidebar.classList.remove('show');
        }
      }
    }
  };

  window.UI = UI;
})();
