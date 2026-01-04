/* static/js/model-selector.js - Complete Model Selector with Enhanced UI */

(function() {
  'use strict';

  // State management
  let currentModelKey = (window.NEXAAI && window.NEXAAI.initialModelKey) || "gemini-flash";
  let currentModelName = (window.NEXAAI && window.NEXAAI.initialModelName) || "Gemini 2.5 Flash";

  // Initialize on page load
  document.addEventListener('DOMContentLoaded', function() {
    initModelSelector();
    updateModelDisplay();
  });

  /**
   * Initialize model selector functionality
   */
  function initModelSelector() {
    const selectorBtn = document.querySelector('.model-selector-btn');
    const selectorDropdown = document.getElementById('model-selector');
    const closeBtn = selectorDropdown?.querySelector('.close-selector');

    // Open/close on button click
    if (selectorBtn) {
      selectorBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        toggleModelSelector();
      });
    }

    // Close button
    if (closeBtn) {
      closeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        closeModelSelector();
      });
    }

    // Close when clicking outside
    document.addEventListener('click', function(e) {
      if (!e.target.closest('.model-selector-btn') && 
          !e.target.closest('#model-selector')) {
        closeModelSelector();
      }
    });

    // Prevent dropdown from closing when clicking inside
    if (selectorDropdown) {
      selectorDropdown.addEventListener('click', function(e) {
        e.stopPropagation();
      });
    }

    // Add hover effects to model cards
    addModelCardEffects();
  }

  /**
   * Toggle model selector visibility
   */
  window.toggleModelSelector = function() {
    const selector = document.getElementById('model-selector');
    if (!selector) return;

    const isVisible = selector.style.display === 'block';

    if (isVisible) {
      closeModelSelector();
    } else {
      openModelSelector();
    }
  };

  /**
   * Open model selector dropdown
   */
  function openModelSelector() {
    const selector = document.getElementById('model-selector');
    if (!selector) return;

    selector.style.display = 'block';
    selector.classList.add('modal-enter');

    // Scroll active model into view
    setTimeout(() => {
      const activeCard = selector.querySelector('.model-card.active');
      if (activeCard) {
        activeCard.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'nearest' 
        });
      }
    }, 100);
  }

  /**
   * Close model selector dropdown
   */
  function closeModelSelector() {
    const selector = document.getElementById('model-selector');
    if (!selector) return;

    selector.classList.remove('modal-enter');
    setTimeout(() => {
      selector.style.display = 'none';
    }, 200);
  }

  /**
   * Select a model
   * @param {string} modelKey - Model identifier
   * @param {string} modelName - Display name of the model
   */
  window.selectModel = async function(modelKey, modelName) {
    if (!modelKey) {
      console.error('Model key is required');
      return;
    }

    // Update state
    currentModelKey = modelKey;
    currentModelName = modelName || modelKey;

    // Update UI
    updateModelDisplay();
    updateActiveModelCard(modelKey);

    // Save to session
    try {
      const response = await fetch('/set-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ 
          model: modelKey,
          name: modelName
        })
      });

      if (!response.ok) {
        throw new Error('Failed to save model selection');
      }

      const data = await response.json();
      console.log('Model selected:', data);

      // Show success indicator
      showModelChangeNotification(modelName);

    } catch (error) {
      console.error('Error selecting model:', error);
      showModelChangeNotification(modelName, true);
    }

    // Close dropdown
    closeModelSelector();
  };

  /**
   * Update model name in UI
   */
  function updateModelDisplay() {
    // Update top nav
    const topModelName = document.getElementById('selected-model-name');
    if (topModelName) {
      topModelName.textContent = currentModelName;
    }

    // Update footer model info
    const footerModelInfo = document.getElementById('model-info');
    if (footerModelInfo) {
      footerModelInfo.textContent = currentModelName;
    }

    // Update any other model displays
    document.querySelectorAll('[data-model-display]').forEach(el => {
      el.textContent = currentModelName;
    });
  }

  /**
   * Update active state of model cards
   * @param {string} modelKey - Selected model key
   */
  function updateActiveModelCard(modelKey) {
    // Remove active class from all cards
    document.querySelectorAll('.model-card').forEach(card => {
      card.classList.remove('active');
    });

    // Add active class to selected card
    const selectedCard = document.querySelector(`[data-model="${CSS.escape(modelKey)}"]`);
    if (selectedCard) {
      selectedCard.classList.add('active');
    }
  }

  /**
   * Add hover and interaction effects to model cards
   */
  function addModelCardEffects() {
    document.querySelectorAll('.model-card').forEach(card => {
      // Highlight on hover
      card.addEventListener('mouseenter', function() {
        if (!this.classList.contains('active')) {
          this.style.borderColor = 'rgba(25, 195, 125, 0.3)';
        }
      });

      card.addEventListener('mouseleave', function() {
        if (!this.classList.contains('active')) {
          this.style.borderColor = '';
        }
      });

      // Add click handler if not already present
      if (!card.onclick && card.dataset.model) {
        card.addEventListener('click', function() {
          const modelKey = this.dataset.model;
          const modelName = this.querySelector('.model-name')?.textContent || modelKey;
          window.selectModel(modelKey, modelName);
        });
      }
    });
  }

  /**
   * Show notification when model changes
   * @param {string} modelName - Name of the model
   * @param {boolean} isError - Whether this is an error notification
   */
  function showModelChangeNotification(modelName, isError = false) {
    const notification = document.createElement('div');
    notification.className = 'model-change-notification';
    notification.style.cssText = `
      position: fixed;
      top: 80px;
      left: 50%;
      transform: translateX(-50%);
      background: ${isError ? '#ef4444' : '#19c37d'};
      color: white;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 500;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 10000;
      animation: slideDown 0.3s ease-out;
    `;
    notification.textContent = isError 
      ? `Failed to switch to ${modelName}` 
      : `Switched to ${modelName}`;

    document.body.appendChild(notification);

    // Remove after 2 seconds
    setTimeout(() => {
      notification.style.animation = 'slideUp 0.3s ease-out';
      setTimeout(() => notification.remove(), 300);
    }, 2000);
  }

  /**
   * Get currently selected model
   * @returns {Object} Current model info
   */
  window.getCurrentModel = function() {
    return {
      key: currentModelKey,
      name: currentModelName
    };
  };

  /**
   * Search/filter models (for future enhancement)
   * @param {string} query - Search query
   */
  window.filterModels = function(query) {
    if (!query) {
      document.querySelectorAll('.model-card').forEach(card => {
        card.style.display = '';
      });
      return;
    }

    const searchLower = query.toLowerCase();

    document.querySelectorAll('.model-card').forEach(card => {
      const modelName = card.querySelector('.model-name')?.textContent.toLowerCase() || '';
      const modelDesc = card.querySelector('.model-description')?.textContent.toLowerCase() || '';

      const matches = modelName.includes(searchLower) || modelDesc.includes(searchLower);
      card.style.display = matches ? '' : 'none';
    });
  };

  /**
   * Group models by tier (free, pro, max)
   */
  window.organizeModelsByTier = function() {
    const selector = document.getElementById('model-selector');
    if (!selector) return;

    const content = selector.querySelector('.model-selector-content');
    if (!content) return;

    const cards = Array.from(content.querySelectorAll('.model-card'));

    // Group by tier
    const tiers = {
      free: [],
      pro: [],
      max: []
    };

    cards.forEach(card => {
      const tier = card.dataset.tier || 'free';
      if (tiers[tier]) {
        tiers[tier].push(card);
      }
    });

    // Render grouped
    content.innerHTML = '';

    Object.entries(tiers).forEach(([tier, tierCards]) => {
      if (tierCards.length === 0) return;

      const group = document.createElement('div');
      group.className = 'model-group';

      const header = document.createElement('div');
      header.className = 'model-group-header';
      header.innerHTML = `
        <span class="group-badge ${tier}">${tier.charAt(0).toUpperCase() + tier.slice(1)}</span>
        <span class="group-title">${tierCards.length} models</span>
      `;

      group.appendChild(header);
      tierCards.forEach(card => group.appendChild(card));

      content.appendChild(group);
    });

    addModelCardEffects();
  };

  // Add CSS animations
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideDown {
      from {
        opacity: 0;
        transform: translate(-50%, -20px);
      }
      to {
        opacity: 1;
        transform: translate(-50%, 0);
      }
    }

    @keyframes slideUp {
      from {
        opacity: 1;
        transform: translate(-50%, 0);
      }
      to {
        opacity: 0;
        transform: translate(-50%, -20px);
      }
    }

    .modal-enter {
      animation: fadeIn 0.2s ease-out;
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: scale(0.95);
      }
      to {
        opacity: 1;
        transform: scale(1);
      }
    }

    .model-card {
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .model-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .model-card.active {
      border-color: #19c37d !important;
      background: rgba(25, 195, 125, 0.05);
    }

    .model-card.active::before {
      content: '✓';
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      width: 20px;
      height: 20px;
      background: #19c37d;
      color: white;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      font-weight: bold;
    }
  `;
  document.head.appendChild(style);

  console.log('✅ Model selector initialized');
})();
