/* global window, document, localStorage */
(function () {
  const Models = {
    init(modelsDict, serverSelected, currentUser) {
      this.models = modelsDict || {};
      this.user = currentUser || { is_premium: false };

      // prefer localStorage selection if exists (client remembers)
      const saved = localStorage.getItem('nexa_selected_model');
      this.selected = saved || serverSelected || 'gemini-flash';

      this.btn = document.getElementById('model-selector-btn');
      this.dd = document.getElementById('model-selector-dropdown');

      this.render();
      this.applySelected(this.selected, false);
    },

    toggleSelector(force) {
      const dd = document.getElementById('model-selector-dropdown');
      if (!dd) return;

      const shouldShow = (typeof force === 'boolean') ? force : !dd.classList.contains('show');
      dd.classList.toggle('show', shouldShow);
      dd.setAttribute('aria-hidden', shouldShow ? 'false' : 'true');
    },

    render() {
      const freeList = document.getElementById('models-free-list');
      const premiumList = document.getElementById('models-premium-list');
      const premiumGroup = document.getElementById('models-premium-group');
      const upgradeCta = document.getElementById('model-upgrade-cta');

      if (!freeList || !premiumList) return;

      freeList.innerHTML = '';
      premiumList.innerHTML = '';

      const entries = Object.entries(this.models);

      const isPremiumUser = !!this.user.is_premium;

      for (const [key, meta] of entries) {
        const isPremiumModel = !meta.limit && key.startsWith('gpt-4') || key.includes('sonnet') || key.includes('opus') || key.includes('gemini-pro') || key.includes('deepseek-r1');
        // safer: if server only sends premium models to premium users, this still works.
        const card = this._makeCard(key, meta);

        if (isPremiumModel) premiumList.appendChild(card);
        else freeList.appendChild(card);
      }

      if (isPremiumUser) {
        if (premiumGroup) premiumGroup.style.display = '';
        if (upgradeCta) upgradeCta.innerHTML = '';
      } else {
        if (premiumGroup) premiumGroup.style.display = 'none';
        if (upgradeCta) {
          upgradeCta.innerHTML = `<div class="upgrade-card" onclick="location.href='/checkout'">
            <div class="upgrade-content">
              <h3>Upgrade to Premium</h3>
              <p>Unlock premium models.</p>
              <button class="upgrade-button">Upgrade</button>
            </div>
          </div>`;
        }
      }
    },

    _makeCard(key, meta) {
      const div = document.createElement('div');
      div.className = 'model-card';
      div.dataset.model = key;
      div.dataset.name = meta.name || key;

      div.innerHTML = `
        <div class="model-card-header">
          <div class="model-info">
            <h4 class="model-name">${meta.name || key}</h4>
            <div class="model-features">
              ${meta.vision ? `<span class="feature-badge vision">Vision</span>` : ``}
              ${meta.limit ? `<span class="feature-badge limit">${meta.limit}/day</span>` : ``}
              <span class="feature-badge speed">Fast</span>
            </div>
          </div>
        </div>
      `;

      div.addEventListener('click', () => {
        this.applySelected(key, true);
        this.toggleSelector(false);
      });

      return div;
    },

    applySelected(modelKey, persist) {
      if (!this.models[modelKey]) modelKey = 'gemini-flash';
      this.selected = modelKey;

      const name = (this.models[modelKey] && this.models[modelKey].name) ? this.models[modelKey].name : modelKey;

      const label1 = document.getElementById('selected-model-name');
      const label2 = document.getElementById('model-info');
      if (label1) label1.textContent = name;
      if (label2) label2.textContent = name;

      // highlight active
      document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
      const active = document.querySelector(`.model-card[data-model="${CSS.escape(modelKey)}"]`);
      if (active) active.classList.add('active');

      if (persist) localStorage.setItem('nexa_selected_model', modelKey);
    },

    getSelected() {
      return this.selected || 'gemini-flash';
    }
  };

  window.Models = Models;
})();
