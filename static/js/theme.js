// static/js/theme.js
(function () {
  const STORAGE_KEY = "theme"; // "system" | "dark" | "light"
  const media = window.matchMedia?.("(prefers-color-scheme: dark)");

  function resolveTheme(pref) {
    if (pref === "dark" || pref === "light") return pref;
    return media && media.matches ? "dark" : "light"; // system
  }

  function applyTheme(themePref) {
    const effective = resolveTheme(themePref);
    document.documentElement.setAttribute("data-theme", effective);
    document.documentElement.setAttribute("data-theme-pref", themePref);
  }

  function setTheme(themePref) {
    localStorage.setItem(STORAGE_KEY, themePref);
    applyTheme(themePref);
    // optional: reflect selection in UI
    document.querySelectorAll("[data-theme-choice]").forEach((el) => {
      el.classList.toggle("active", el.dataset.themeChoice === themePref);
    });
  }

  function initTheme() {
    const saved = localStorage.getItem(STORAGE_KEY) || "system";
    applyTheme(saved);

    // If system theme changes and user pref is "system", update automatically
    if (media) {
      media.addEventListener?.("change", () => {
        const pref = localStorage.getItem(STORAGE_KEY) || "system";
        if (pref === "system") applyTheme(pref);
      });
    }

    // Click bindings for theme buttons (use data-theme-choice="dark|light|system")
    document.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-theme-choice]");
      if (!btn) return;
      setTheme(btn.dataset.themeChoice);
    });
  }

  window.NexaTheme = { initTheme, setTheme };
})();
