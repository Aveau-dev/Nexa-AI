/* static/js/settings.js
   Settings & integrations helpers
*/

const Settings = (function () {
  async function saveAccount() {
    const name = document.getElementById("settings-name").value;
    try {
      const resp = await fetch("/api/settings/account", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name })
      });
      const data = await resp.json();
      if (data.error) UI.toast("Save failed");
      else UI.toast("Account updated");
    } catch (e) {
      console.error("Settings.saveAccount error", e);
      UI.toast("Save failed");
    }
  }

  function connectGoogle() {
    window.location.href = "/integrations/google/start";
  }

  function connectDrive() {
    window.location.href = "/integrations/drive/start";
  }

  function connectNotion() {
    window.location.href = "/integrations/notion/start";
  }

  return {
    saveAccount,
    connectGoogle,
    connectDrive,
    connectNotion
  };
})();

window.Settings = Settings;
