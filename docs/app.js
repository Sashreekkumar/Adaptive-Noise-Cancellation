/**
 * noisygen Documentation Application Runtime
 * Clean, fast client-side router, search engine, CLI builder, and theme manager.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    currentSectionId: 'overview',
    theme: localStorage.getItem('noisygen_theme') || 'light',
    searchQuery: '',
    sidebarFilter: '',
    builder: {
      root: '/data/noisygen',
      output: '/data/noisygen/output',
      config: 'config.json',
      numSamples: 10000,
      seed: 42,
      workers: 8,
      probeAll: true,
      cacheDir: '.cache',
      resume: false,
      overwrite: false,
      summaryOnly: false,
      reconstruct: ''
    }
  };

  let el = {};

  function init() {
    cacheElements();
    applyTheme(state.theme);
    renderSidebar();
    setupEventListeners();
    setupRouting();
    initCliBuilder();
    initReadingProgress();
  }

  function cacheElements() {
    el = {
      sidebarNav: document.getElementById('sidebar-nav'),
      sidebarFilterInput: document.getElementById('sidebar-filter-input'),
      docContentContainer: document.getElementById('doc-content-container'),
      tocContainer: document.getElementById('toc-container'),
      breadcrumbsContainer: document.getElementById('breadcrumbs-container'),
      readingProgressBar: document.getElementById('reading-progress-bar'),
      themeToggleBtn: document.getElementById('theme-toggle-btn'),
      mobileMenuBtn: document.getElementById('mobile-menu-btn'),
      appSidebar: document.getElementById('app-sidebar'),
      globalSearchTrigger: document.getElementById('global-search-trigger'),
      searchModal: document.getElementById('search-modal'),
      searchInput: document.getElementById('search-input'),
      searchResultsList: document.getElementById('search-results-list'),
      cliBuilderModal: document.getElementById('cli-builder-modal'),
      toastContainer: document.getElementById('toast-container')
    };
  }

  // ==========================================================================
  // Routing & Section Rendering
  // ==========================================================================
  function setupRouting() {
    window.addEventListener('hashchange', handleHashChange);
    handleHashChange();
  }

  function handleHashChange() {
    let hash = window.location.hash.replace('#', '').trim();
    if (!hash) {
      hash = 'overview';
    }

    const exactSection = window.DOCS_DATA.find(s => s.id === hash);
    if (exactSection) {
      loadSection(exactSection.id);
      window.scrollTo({ top: 0, behavior: 'auto' });
    } else {
      const parentSection = window.DOCS_DATA.find(s => {
        return hash.startsWith(s.id) || (s.toc && s.toc.some(t => t.id === hash));
      });

      if (parentSection) {
        loadSection(parentSection.id, hash);
      } else {
        loadSection('overview');
      }
    }
  }

  function loadSection(sectionId, anchorId = null) {
    const sec = window.DOCS_DATA.find(s => s.id === sectionId);
    if (!sec) return;

    state.currentSectionId = sectionId;

    updateActiveSidebarItem(sectionId);
    renderBreadcrumbs(sec);
    renderMainContent(sec);
    renderToc(sec);

    document.title = `${sec.title} — noisygen Documentation`;

    if (anchorId) {
      setTimeout(() => {
        const targetEl = document.getElementById(anchorId);
        if (targetEl) {
          targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 50);
    }
  }

  function renderBreadcrumbs(sec) {
    if (!el.breadcrumbsContainer) return;
    el.breadcrumbsContainer.innerHTML = `
      <a href="#overview" class="crumb-link">Documentation</a>
      <span class="crumb-sep">/</span>
      <span class="crumb-group">${sec.group}</span>
      <span class="crumb-sep">/</span>
      <span class="crumb-current">${sec.title}</span>
    `;
  }

  function renderMainContent(sec) {
    if (!el.docContentContainer) return;

    const currentIndex = window.DOCS_DATA.findIndex(s => s.id === sec.id);
    const prevSec = currentIndex > 0 ? window.DOCS_DATA[currentIndex - 1] : null;
    const nextSec = currentIndex < window.DOCS_DATA.length - 1 ? window.DOCS_DATA[currentIndex + 1] : null;

    const navButtonsHtml = `
      <div class="bottom-page-nav">
        ${prevSec ? `
          <a href="#${prevSec.id}" class="page-nav-link prev">
            <span class="nav-hint">&larr; Previous</span>
            <span class="nav-target">${prevSec.title}</span>
          </a>
        ` : '<div></div>'}
        ${nextSec ? `
          <a href="#${nextSec.id}" class="page-nav-link next">
            <span class="nav-hint">Next &rarr;</span>
            <span class="nav-target">${nextSec.title}</span>
          </a>
        ` : ''}
      </div>
    `;

    el.docContentContainer.innerHTML = `
      <header class="section-doc-header">
        <div class="section-tag">${sec.num === 0 ? 'Package Overview' : `Section ${sec.num}`}</div>
        <h1 class="section-doc-title">${sec.title}</h1>
      </header>
      <div class="section-doc-body">
        ${sec.html}
      </div>
      ${navButtonsHtml}
    `;

    if (el.appSidebar && el.appSidebar.classList.contains('mobile-open')) {
      el.appSidebar.classList.remove('mobile-open');
    }
  }

  function renderToc(sec) {
    if (!el.tocContainer) return;
    if (!sec.toc || sec.toc.length === 0) {
      el.tocContainer.innerHTML = `
        <div class="toc-title">On this page</div>
        <div class="toc-empty">Overview &amp; references</div>
        <div class="toc-footer">
          <a class="toc-footer-link" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">Back to top &uarr;</a>
          <a class="toc-footer-link" onclick="copyCurrentUrl()">Copy page URL</a>
        </div>
      `;
      return;
    }

    const items = sec.toc.map(item => `
      <li>
        <a href="#${item.id}" class="toc-item ${item.level === 3 ? 'level-sub' : ''}">
          ${item.text}
        </a>
      </li>
    `).join('');

    el.tocContainer.innerHTML = `
      <div class="toc-title">On this page</div>
      <ul class="toc-items-list">
        ${items}
      </ul>
      <div class="toc-footer">
        <a class="toc-footer-link" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">Back to top &uarr;</a>
        <a class="toc-footer-link" onclick="copyCurrentUrl()">Copy page URL</a>
      </div>
    `;

    setupTocScrollSpy();
  }

  function setupTocScrollSpy() {
    const headingAnchors = el.docContentContainer.querySelectorAll('.standard-h2, .standard-h3');
    if (!headingAnchors.length) return;

    window.addEventListener('scroll', () => {
      const scrollPos = window.scrollY + 80;
      let currentAnchor = null;

      headingAnchors.forEach(h => {
        if (h.offsetTop <= scrollPos) {
          currentAnchor = h.id;
        }
      });

      if (currentAnchor && el.tocContainer) {
        const links = el.tocContainer.querySelectorAll('.toc-item');
        links.forEach(l => {
          if (l.getAttribute('href') === `#${currentAnchor}`) {
            l.classList.add('active');
          } else {
            l.classList.remove('active');
          }
        });
      }
    }, { passive: true });
  }

  // ==========================================================================
  // Sidebar Rendering & Filtering
  // ==========================================================================
  function renderSidebar() {
    if (!el.sidebarNav) return;

    const groups = {};
    window.DOCS_DATA.forEach(sec => {
      if (!groups[sec.group]) {
        groups[sec.group] = {
          name: sec.group,
          order: sec.groupOrder,
          items: []
        };
      }
      groups[sec.group].items.push(sec);
    });

    const sortedGroups = Object.values(groups).sort((a, b) => a.order - b.order);

    let html = '';
    sortedGroups.forEach(grp => {
      const filteredItems = grp.items.filter(item => {
        if (!state.sidebarFilter) return true;
        const q = state.sidebarFilter.toLowerCase();
        return item.title.toLowerCase().includes(q) || item.num.toString().includes(q);
      });

      if (filteredItems.length === 0 && state.sidebarFilter) {
        return;
      }

      html += `
        <div class="sidebar-group" data-group="${grp.name}">
          <div class="sidebar-group-title" onclick="toggleNavGroup(this)">
            <span>${grp.name}</span>
            <svg class="group-arrow" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </div>
          <ul class="sidebar-items">
            ${filteredItems.map(item => `
              <li>
                <a href="#${item.id}" class="sidebar-link ${item.id === state.currentSectionId ? 'active' : ''}" data-id="${item.id}">
                  <span class="link-number">${item.num === 0 ? '' : `${item.num}.`}</span>
                  <span class="link-title">${item.title}</span>
                </a>
              </li>
            `).join('')}
          </ul>
        </div>
      `;
    });

    el.sidebarNav.innerHTML = html;
  }

  function updateActiveSidebarItem(sectionId) {
    if (!el.sidebarNav) return;
    const links = el.sidebarNav.querySelectorAll('.sidebar-link');
    links.forEach(l => {
      if (l.dataset.id === sectionId) {
        l.classList.add('active');
        const groupEl = l.closest('.sidebar-group');
        if (groupEl) groupEl.classList.remove('collapsed');
      } else {
        l.classList.remove('active');
      }
    });
  }

  window.toggleNavGroup = function (headerEl) {
    const group = headerEl.closest('.sidebar-group');
    if (group) {
      group.classList.toggle('collapsed');
    }
  };

  // ==========================================================================
  // Reading Progress Bar
  // ==========================================================================
  function initReadingProgress() {
    window.addEventListener('scroll', () => {
      if (!el.readingProgressBar) return;
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = height > 0 ? (winScroll / height) * 100 : 0;
      el.readingProgressBar.style.width = scrolled + '%';
    }, { passive: true });
  }

  // ==========================================================================
  // Theme Switching
  // ==========================================================================
  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('noisygen_theme', theme);

    if (el.themeToggleBtn) {
      if (theme === 'light') {
        el.themeToggleBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
        `;
        el.themeToggleBtn.title = "Switch to Dark Mode";
      } else {
        el.themeToggleBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
        `;
        el.themeToggleBtn.title = "Switch to Light Mode";
      }
    }
  }

  function toggleTheme() {
    const nextTheme = state.theme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  }

  // ==========================================================================
  // Full-Text Instant Search Modal
  // ==========================================================================
  function openSearchModal() {
    if (!el.searchModal) return;
    el.searchModal.classList.add('open');
    if (el.searchInput) {
      el.searchInput.value = '';
      el.searchInput.focus();
      executeSearch('');
    }
  }

  function closeSearchModal() {
    if (!el.searchModal) return;
    el.searchModal.classList.remove('open');
  }

  function executeSearch(query) {
    if (!el.searchResultsList) return;
    query = query.trim().toLowerCase();

    if (!query) {
      const defaultDocs = window.DOCS_DATA.slice(0, 7);
      renderSearchResults(defaultDocs, '');
      return;
    }

    const matches = [];
    window.DOCS_DATA.forEach(sec => {
      let score = 0;
      const titleLower = sec.title.toLowerCase();
      const plainLower = sec.plainText.toLowerCase();

      if (titleLower === query) score += 100;
      else if (titleLower.includes(query)) score += 50;
      else if (plainLower.includes(query)) score += 20;

      if (sec.toc && sec.toc.some(t => t.text.toLowerCase().includes(query))) {
        score += 35;
      }

      if (score > 0) {
        matches.push({ sec, score, query });
      }
    });

    matches.sort((a, b) => b.score - a.score);
    renderSearchResults(matches.map(m => m.sec), query);
  }

  function renderSearchResults(results, query) {
    if (results.length === 0) {
      el.searchResultsList.innerHTML = `
        <div class="search-empty-state">
          <p>No results found for "<strong>${escapeHtml(query)}</strong>"</p>
        </div>
      `;
      return;
    }

    const items = results.slice(0, 10).map(sec => {
      let snippet = sec.summary;
      if (query && sec.plainText) {
        const idx = sec.plainText.toLowerCase().indexOf(query.toLowerCase());
        if (idx !== -1) {
          const start = Math.max(0, idx - 40);
          const end = Math.min(sec.plainText.length, idx + 100);
          snippet = (start > 0 ? '...' : '') + sec.plainText.substring(start, end) + '...';
          snippet = highlightSnippet(snippet, query);
        }
      }

      return `
        <a href="#${sec.id}" class="search-result-row" onclick="closeSearchModal()">
          <div class="result-header">
            <span class="result-title">${sec.title}</span>
            <span class="result-group">${sec.group}</span>
          </div>
          <div class="result-snippet">${snippet}</div>
        </a>
      `;
    }).join('');

    el.searchResultsList.innerHTML = items;
  }

  function highlightSnippet(text, query) {
    if (!query) return escapeHtml(text);
    const escaped = escapeHtml(text);
    const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
    return escaped.replace(regex, '<mark class="search-mark">$1</mark>');
  }

  // ==========================================================================
  // CLI Command & Config Generator Modal
  // ==========================================================================
  function initCliBuilder() {
    updateBuilderPreview();
  }

  window.openCliBuilder = function () {
    if (!el.cliBuilderModal) return;
    el.cliBuilderModal.classList.add('open');
    updateBuilderPreview();
  };

  window.closeCliBuilder = function () {
    if (!el.cliBuilderModal) return;
    el.cliBuilderModal.classList.remove('open');
  };

  window.applyPreset = function (presetName) {
    if (presetName === 'smoke') {
      state.builder.numSamples = 100;
      state.builder.workers = 2;
      state.builder.seed = 42;
      state.builder.probeAll = false;
      state.builder.resume = false;
    } else if (presetName === 'prod') {
      state.builder.numSamples = 10000;
      state.builder.workers = 8;
      state.builder.seed = 1337;
      state.builder.probeAll = true;
      state.builder.resume = true;
    } else if (presetName === 'clean-probe') {
      state.builder.summaryOnly = true;
      state.builder.probeAll = true;
    } else if (presetName === 'reconstruct') {
      state.builder.reconstruct = 'output/metadata/dataset_metadata.jsonl';
    }

    syncBuilderInputs();
    updateBuilderPreview();
  };

  function syncBuilderInputs() {
    const setVal = (id, val) => {
      const input = document.getElementById(id);
      if (input) {
        if (input.type === 'checkbox') input.checked = !!val;
        else input.value = val;
      }
    };

    setVal('bld-root', state.builder.root);
    setVal('bld-output', state.builder.output);
    setVal('bld-config', state.builder.config);
    setVal('bld-samples', state.builder.numSamples);
    setVal('bld-seed', state.builder.seed);
    setVal('bld-workers', state.builder.workers);
    setVal('bld-probe-all', state.builder.probeAll);
    setVal('bld-resume', state.builder.resume);
    setVal('bld-overwrite', state.builder.overwrite);
    setVal('bld-summary-only', state.builder.summaryOnly);
  }

  window.onBuilderInputChange = function () {
    const getVal = (id) => {
      const input = document.getElementById(id);
      if (!input) return null;
      return input.type === 'checkbox' ? input.checked : input.value;
    };

    state.builder.root = getVal('bld-root') || '.';
    state.builder.output = getVal('bld-output') || './output';
    state.builder.config = getVal('bld-config') || 'config.json';
    state.builder.numSamples = parseInt(getVal('bld-samples'), 10) || 1000;
    state.builder.seed = parseInt(getVal('bld-seed'), 10) || 42;
    state.builder.workers = parseInt(getVal('bld-workers'), 10) || 4;
    state.builder.probeAll = !!getVal('bld-probe-all');
    state.builder.resume = !!getVal('bld-resume');
    state.builder.overwrite = !!getVal('bld-overwrite');
    state.builder.summaryOnly = !!getVal('bld-summary-only');

    updateBuilderPreview();
  };

  function updateBuilderPreview() {
    const b = state.builder;
    let parts = ['python -m noisygen'];

    if (b.root && b.root !== '.') parts.push(`--root "${b.root}"`);
    if (b.output && b.output !== './output') parts.push(`--output "${b.output}"`);
    if (b.config && b.config !== 'config.json') parts.push(`--config "${b.config}"`);
    if (b.numSamples) parts.push(`--num-samples ${b.numSamples}`);
    if (b.seed) parts.push(`--seed ${b.seed}`);
    if (b.workers) parts.push(`--workers ${b.workers}`);
    if (b.probeAll) parts.push(`--probe-all`);
    if (b.resume) parts.push(`--resume`);
    if (b.overwrite) parts.push(`--overwrite`);
    if (b.summaryOnly) parts.push(`--summary-only`);
    if (b.reconstruct) parts.push(`--reconstruct "${b.reconstruct}"`);

    const commandStr = parts.join(' \\\n    ');
    const cmdEl = document.getElementById('builder-cmd-code');
    if (cmdEl) {
      cmdEl.textContent = commandStr;
    }

    const yamlStr = `# noisygen configuration
splits:
  train:
    num_samples: ${Math.round(b.numSamples * 0.8)}
    seed: ${b.seed}
  validation:
    num_samples: ${Math.round(b.numSamples * 0.1)}
    seed: ${b.seed + 1}
  test:
    num_samples: ${Math.round(b.numSamples * 0.1)}
    seed: ${b.seed + 2}

target_snr_db:
  min: -5.0
  max: 20.0
snr_method: "active_rms"
clipping_policy: "scale"
sample_rate: 48000
channels: 1
output_subtype: "PCM_16"
`;
    const yamlEl = document.getElementById('builder-yaml-code');
    if (yamlEl) {
      yamlEl.textContent = yamlStr;
    }
  }

  window.switchBuilderTab = function (tabName) {
    const cmdPanel = document.getElementById('builder-cmd-panel');
    const yamlPanel = document.getElementById('builder-yaml-panel');
    const tabs = document.querySelectorAll('.builder-tab');

    tabs.forEach(t => t.classList.remove('active'));

    if (tabName === 'cmd') {
      if (cmdPanel) cmdPanel.style.display = 'block';
      if (yamlPanel) yamlPanel.style.display = 'none';
      if (tabs[0]) tabs[0].classList.add('active');
    } else {
      if (cmdPanel) cmdPanel.style.display = 'none';
      if (yamlPanel) yamlPanel.style.display = 'block';
      if (tabs[1]) tabs[1].classList.add('active');
    }
  };

  // ==========================================================================
  // Clipboard Copy & Notifications
  // ==========================================================================
  window.copyCode = function (button) {
    const wrapper = button.closest('.standard-code-block');
    if (!wrapper) return;
    const code = wrapper.querySelector('code');
    if (!code) return;

    navigator.clipboard.writeText(code.innerText).then(() => {
      button.classList.add('copied');
      const textSpan = button.querySelector('.copy-text');
      if (textSpan) textSpan.textContent = 'Copied';

      setTimeout(() => {
        button.classList.remove('copied');
        if (textSpan) textSpan.textContent = 'Copy';
      }, 1500);
    });
  };

  window.copySnippet = function (elementId) {
    const target = document.getElementById(elementId);
    if (!target) return;
    navigator.clipboard.writeText(target.textContent).then(() => {
      showToast('Copied to clipboard');
    });
  };

  window.copyCurrentUrl = function () {
    navigator.clipboard.writeText(window.location.href).then(() => {
      showToast('URL copied to clipboard');
    });
  };

  window.jumpToSection = function (sectionId) {
    window.location.hash = sectionId;
  };

  function showToast(message) {
    if (!el.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'official-toast';
    toast.textContent = message;
    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.remove();
    }, 2000);
  }

  // ==========================================================================
  // Global Event Listeners
  // ==========================================================================
  function setupEventListeners() {
    if (el.themeToggleBtn) {
      el.themeToggleBtn.addEventListener('click', toggleTheme);
    }

    if (el.mobileMenuBtn && el.appSidebar) {
      el.mobileMenuBtn.addEventListener('click', () => {
        el.appSidebar.classList.toggle('mobile-open');
      });
    }

    if (el.sidebarFilterInput) {
      el.sidebarFilterInput.addEventListener('input', (e) => {
        state.sidebarFilter = e.target.value;
        renderSidebar();
      });
    }

    if (el.globalSearchTrigger) {
      el.globalSearchTrigger.addEventListener('click', openSearchModal);
    }

    if (el.searchInput) {
      el.searchInput.addEventListener('input', (e) => {
        executeSearch(e.target.value);
      });
    }

    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        openSearchModal();
      }

      if (e.key === 'Escape') {
        closeSearchModal();
        closeCliBuilder();
      }
    });

    document.querySelectorAll('.modal-backdrop').forEach(modal => {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          modal.classList.remove('open');
        }
      });
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
