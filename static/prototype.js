/**
 * INTELLIGENT JOB SCHEDULER - EDITORIAL MEDICAL PROTOTYPE SCRIPT
 * Features:
 * 1. Smooth Section Navigation (Home, About, Contact)
 * 2. Adaptive Login Route Resolution (file:// and http://)
 * 3. Interactive Clinical ER Queue Dispatcher Simulation
 * 4. Modal Viewer & Toast Notification System
 */

(function () {
  'use strict';

  // =========================================================================
  // 1. SECTION NAVIGATION & LOGIN LINK ADAPTER
  // =========================================================================
  function initNavScroll() {
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('section[id]');

    function updateActiveNav() {
      const scrollPos = window.scrollY + 120;
      sections.forEach(sec => {
        const top = sec.offsetTop;
        const height = sec.offsetHeight;
        const id = sec.getAttribute('id');
        if (scrollPos >= top && scrollPos < top + height) {
          navItems.forEach(item => {
            item.classList.remove('active');
            const link = item.querySelector('a');
            if (link && link.getAttribute('href') === `#${id}`) {
              item.classList.add('active');
            }
          });
        }
      });
    }

    window.addEventListener('scroll', updateActiveNav, { passive: true });
    updateActiveNav();

    // Adapt Login buttons for both file:// and http:// protocols
    const isFileProtocol = window.location.protocol === 'file:';
    document.querySelectorAll('#loginButton, a[href="login.html"], a[href="/login"]').forEach(btn => {
      btn.setAttribute('href', isFileProtocol ? 'login.html' : '/login');
    });
  }

  // =========================================================================
  // 2. INTERACTIVE CLINICAL ER QUEUE DISPATCHER SIMULATION
  // =========================================================================
  const samplePatients = [
    { id: 'PT-1042', arrival: '08:14', acuity: 1, label: 'Resuscitation', complaint: 'Acute STEMI / Ventricular Tachycardia', doctor: 'Dr. Ruoyu Li', status: 'In Trauma Bay 1' },
    { id: 'PT-1043', arrival: '08:22', acuity: 2, label: 'Emergent', complaint: 'Severe Respiratory Distress (SpO2 84%)', doctor: 'Dr. Yifan Wang', status: 'Allocated' },
    { id: 'PT-1044', arrival: '08:35', acuity: 2, label: 'Emergent', complaint: 'Suspected Ischemic Stroke (NIHSS 14)', doctor: 'Dr. Housen Zhu', status: 'CT Queue' },
    { id: 'PT-1045', arrival: '08:48', acuity: 3, label: 'Urgent', complaint: 'Compound Tibial Fracture w/ Compartment Check', doctor: 'Dr. Yuan Sui', status: 'Waiting' },
    { id: 'PT-1046', arrival: '09:05', acuity: 3, label: 'Urgent', complaint: 'Acute Abdomen / RLQ Peritoneal Signs', doctor: 'Dr. Ruoyu Li', status: 'Waiting' },
    { id: 'PT-1047', arrival: '09:15', acuity: 4, label: 'Less Urgent', complaint: 'Laceration 4cm Forearm (Active Bleeding Controlled)', doctor: 'Triage Pool', status: 'Waiting' },
    { id: 'PT-1048', arrival: '09:20', acuity: 5, label: 'Non-urgent', complaint: 'Subacute Phalangeal Sprain', doctor: 'Triage Pool', status: 'Waiting' }
  ];

  let currentQueue = [...samplePatients];
 
  function renderQueueTable() {
    const tbody = document.getElementById('queueTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';
    currentQueue.forEach((pt, index) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>#${index + 1}</strong> <span style="font-family:var(--font-mono);">${pt.id}</span></td>
        <td>${pt.arrival}</td>
        <td><span class="acuity-tag acuity-${pt.acuity}">${pt.acuity} · ${pt.label}</span></td>
        <td>${pt.complaint}</td>
        <td><strong>${pt.doctor}</strong></td>
        <td><span class="glass-badge ${pt.acuity <= 2 ? 'badge-danger' : 'badge-success'}">${pt.status}</span></td>
      `;
      tbody.appendChild(tr);
    });

    const queueCounter = document.getElementById('activeQueueCount');
    if (queueCounter) {
      queueCounter.textContent = `${currentQueue.length} Active Cases`;
    }
  }

  function initQueueSimulator() {
    renderQueueTable();
  }

  // =========================================================================
  // 3. CONTACT FORM & INTERACTIVE TOAST
  // =========================================================================
  function showToast(message) {
    let container = document.getElementById('toastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toastContainer';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(-10px)';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  function initContact() {
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
      contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        showToast('✉️ Submission received! The Editorial Secretariat will reply within 24 hours.');
        contactForm.reset();
      });
    }
  }

  // =========================================================================
  // 4. ARTICLE MODAL VIEWER
  // =========================================================================
  const articleDatabase = {
    'allocation': {
      title: 'Autonomous Clinical Workload Balancing via Multi-Agent Optimization',
      tag: 'Clinical Trial · Phase III',
      doi: '10.1016/j.medsched.2025.04.012',
      authors: 'Yifan Wang, Housen Zhu, Ruoyu Li, Yuan Sui',
      body: `
        <h3>Executive Abstract</h3>
        <p>In high-density level-1 trauma centers, asynchronous patient arrivals present severe queue instability and clinician cognitive overload. This study presents a dynamic multi-agent job scheduler built upon Markov Decision Processes (MDP) and Hungarian bipartite matching.</p>
        <div class="math-formula-box">
          <div class="math-formula-expr">
            <span class="math-cal">R</span>(<span class="math-var">s</span>, <span class="math-var">a</span>)
            <span class="math-op">=</span>
            <span class="math-op">−</span>
            <span style="font-size:1.3rem; margin-right: 2px;">∑</span><sub class="math-sub">i ∈ <span class="math-cal">Q</span></sub>
            <span class="math-greek">ω</span><sub class="math-sub">i</sub>
            <span class="math-op">·</span>
            (<span class="math-var">t</span><sub class="math-sub">wait, i</sub>)<sup class="math-sup"><span class="math-greek">γ</span></sup>
            <span class="math-op">−</span>
            <span class="math-greek">λ</span>
            <span class="math-op">·</span>
            <span class="math-var">σ</span>(<strong class="math-bold">W</strong><sub class="math-sub">doctors</sub>)
          </div>
        </div>
        <p>Our clinical deployment across 48,000 emergency admissions demonstrated a 41.8% reduction in high-acuity wait times and eliminated clinical bottleneck hotspots without requiring additional staffing overhead.</p>
      `
    },
    'queuing': {
      title: 'Dynamic Queue Priority Rebalancing Under Surge Conditions',
      tag: 'Methodology & Algorithm',
      doi: '10.1016/j.medsched.2025.03.008',
      authors: 'Housen Zhu, MS; Yuan Sui, MS; Ruoyu Li, MD',
      body: `
        <h3>Key Findings</h3>
        <p>When surge alerts are triggered (mass casualty incidents or seasonal epidemics), static FIFO queues fail catastrophically. Our system introduces dynamic acuity re-weighting with automated telemetry feedback.</p>
        <p>Real-time vital sign feeds adjust the priority coefficient dynamically, guaranteeing resuscitation cases immediate bed allocation within sub-second latency.</p>
      `
    }
  };

  function renderMath(targetElement) {
    const el = targetElement || document.body;
    if (window.renderMathInElement) {
      window.renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false
      });
    }
  }

  function initMathRendering() {
    renderMath();
    window.addEventListener('load', () => renderMath());
    setTimeout(() => renderMath(), 250);
    setTimeout(() => renderMath(), 800);
  }

  function initModalViewer() {
    const modalOverlay = document.getElementById('articleModalOverlay');
    const modalContent = document.getElementById('modalArticleContent');
    const modalClose = document.getElementById('modalCloseBtn');

    if (!modalOverlay || !modalContent) return;

    document.querySelectorAll('[data-open-article]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const articleKey = btn.getAttribute('data-open-article') || 'allocation';
        const article = articleDatabase[articleKey] || articleDatabase['allocation'];

        modalContent.innerHTML = `
          <div class="lead-meta-row" style="margin-bottom: 12px;">
            <span class="glass-badge badge-success">${article.tag}</span>
            <span class="lead-doi">DOI: ${article.doi}</span>
          </div>
          <h2 style="font-size: 1.8rem; margin-bottom: 12px;">${article.title}</h2>
          <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 20px;">${article.authors}</p>
          <div class="editorial-divider" style="margin: 16px 0;"><span>EDITORIAL TRANSCRIPT</span></div>
          <div class="article-rich-text">${article.body}</div>
          <div style="margin-top: 24px; display: flex; gap: 10px;">
            <button class="btn-primary" onclick="window.print()">Print Paper</button>
            <button class="btn-outline" id="citePaperBtn">Copy BibTeX Citation</button>
          </div>
        `;

        renderMath(modalContent);
        modalOverlay.classList.add('open');

        const citeBtn = document.getElementById('citePaperBtn');
        if (citeBtn) {
          citeBtn.addEventListener('click', () => {
            const bibtex = `@article{li2025scheduler,\n  title={${article.title}},\n  author={${article.authors}},\n  journal={MedSched Intelligence},\n  year={2025},\n  doi={${article.doi}}\n}`;
            navigator.clipboard.writeText(bibtex).then(() => {
              showToast('📚 BibTeX citation copied!');
            });
          });
        }
      });
    });

    if (modalClose) {
      modalClose.addEventListener('click', () => {
        modalOverlay.classList.remove('open');
      });
    }

    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) {
        modalOverlay.classList.remove('open');
      }
    });
  }

  // =========================================================================
  // INITIALIZATION ON DOM READY
  // =========================================================================
  document.addEventListener('DOMContentLoaded', () => {
    initNavScroll();
    initQueueSimulator();
    initContact();
    initModalViewer();
    initMathRendering();
    console.log('🏛️ MedSched Intelligence Initialized');
  });

})();
