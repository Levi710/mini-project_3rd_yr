/* ══════════════════════════════════════════════════════════════
   Pluto v1.0 Dashboard — app.js
   SSE listener + dynamic rendering for pipeline dashboard
   ══════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── DOM refs ────────────────────────────────────────────────
  const queryInput = document.getElementById('queryInput');
  const runBtn = document.getElementById('runBtn');
  const answerBody = document.getElementById('answerBody');
  const evidenceBody = document.getElementById('evidenceBody');
  const traceBody = document.getElementById('traceBody');
  const confRing = document.getElementById('confRing');
  const confValue = document.getElementById('confValue');

  const stages = ['route', 'extract', 'merge', 'verify'];
  const stageEls = {};
  const statusEls = {};
  const connectors = document.querySelectorAll('.stage-rail__connector');

  stages.forEach(s => {
    stageEls[s] = document.getElementById(`stage-${s}`);
    statusEls[s] = document.getElementById(`status-${s}`);
  });

  // ── Run pipeline ────────────────────────────────────────────
  runBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    if (!query) return;

    runBtn.disabled = true;
    runBtn.innerHTML = '<span class="spinner"></span> Running…';
    resetUI();

    // Start SSE listener FIRST — wait for connection to fully open before sending request
    await listenSSE();

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      let data;
      try {
        data = await res.json();
      } catch (parseErr) {
        throw new Error(`Server returned an invalid response (${res.status} ${res.statusText})`);
      }

      if (!res.ok) {
        throw new Error(data.error || `Server error: ${res.status}`);
      }

      renderResult(data);
    } catch (err) {
      answerBody.innerHTML = `<div style="padding:16px;background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.3);border-radius:8px;color:var(--accent-red);">
        <strong>Pipeline Error:</strong> ${err.message}
      </div>`;
      console.error(err);
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = '<span class="btn-icon">▶</span> Run Pipeline';
    }
  });

  // Enter key support
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && queryInput.value.trim()) runBtn.click();
  });

  // Enable/disable Run button based on query input
  queryInput.addEventListener('input', () => {
    const hasText = queryInput.value.trim().length > 0;
    runBtn.disabled = !hasText;
    runBtn.style.opacity = hasText ? '1' : '0.5';
    runBtn.style.cursor = hasText ? '' : 'not-allowed';
  });

  // ── SSE stream ──────────────────────────────────────────────
  function listenSSE() {
    return new Promise((resolve, reject) => {
      const es = new EventSource('/api/stream');

      es.onopen = () => {
        resolve(es);
      };

      es.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          handleProgress(data);
          if (data.stage === 'done') es.close();
        } catch (_) { }
      };

      es.onerror = (err) => {
        // Only reject if it errors before opening
        if (es.readyState === EventSource.CONNECTING) {
          reject(err);
        }
        es.close();
      };
    });
  }

  function handleProgress(data) {
    const stage = data.stage;

    // Handle Phase A: understand stage (not in the visual pipeline stages)
    if (stage === 'understand') {
      const status = data.status;
      if (status === 'running') {
        answerBody.innerHTML = '<p class="panel__placeholder"><span class="spinner"></span> Phase A: Understanding document…</p>';
      } else if (status === 'complete') {
        answerBody.innerHTML = '<p class="panel__placeholder"><span class="spinner"></span> Document understood. Running query…</p>';
      }
      return;
    }

    if (!stages.includes(stage) && stage !== 'finish' && stage !== 'done') return;

    if (stage === 'finish' || stage === 'done') {
      // Mark all stages complete
      stages.forEach((s, i) => {
        stageEls[s].classList.remove('active');
        stageEls[s].classList.add('complete');
        statusEls[s].textContent = 'done';
        if (connectors[i]) connectors[i].classList.add('active');
      });
      return;
    }

    const status = data.status;
    const idx = stages.indexOf(stage);

    if (status === 'running') {
      stageEls[stage].classList.add('active');
      stageEls[stage].classList.remove('complete');
      statusEls[stage].innerHTML = '<span class="status-dot status-dot--running"></span>running';
      // Update answer body to show current stage
      answerBody.innerHTML = `<p class="panel__placeholder"><span class="spinner"></span> ${stage.toUpperCase()}: processing…</p>`;
    } else if (status === 'complete') {
      stageEls[stage].classList.remove('active');
      stageEls[stage].classList.add('complete');

      // Show extra info if available
      let info = 'done';
      if (stage === 'route' && data.chunks) info = `done (${data.chunks} chunks)`;
      if (stage === 'extract' && data.extractions) info = `done (${data.extractions} facts)`;
      if (stage === 'merge' && data.key_claims) info = `done (${data.key_claims} claims)`;
      if (stage === 'verify' && data.checked) info = `done (${data.checked} verified)`;

      statusEls[stage].innerHTML = `<span class="status-dot status-dot--complete"></span>${info}`;
      // Light up connector
      if (idx < connectors.length) connectors[idx].classList.add('active');
    }
  }

  // ── Render result ───────────────────────────────────────────
  function renderResult(data) {
    if (data.error) {
      answerBody.innerHTML = `<p style="color:var(--accent-red);">${data.error}</p>`;
      return;
    }

    // Answer sections
    const fa = data.final_answer || {};
    let html = '';

    if (fa.sections && fa.sections.length) {
      fa.sections.forEach(sec => {
        html += `
          <div class="answer-section animate-in">
            <div class="answer-section__title">${esc(sec.title)}</div>
            <div class="answer-section__content">${esc(sec.content)}</div>
          </div>`;
      });
    } else if (fa.response) {
      html = `<div class="answer-section animate-in">
        <div class="answer-section__content">${esc(fa.response)}</div>
      </div>`;
    }
    answerBody.innerHTML = html || '<p class="panel__placeholder">No answer generated</p>';

    // Evidence
    const ev = data.evidence || [];
    if (ev.length) {
      evidenceBody.innerHTML = ev.map(e => `
        <div class="evidence-card animate-in">
          <div class="evidence-card__source">${esc(e.doc_id)} / ${esc(e.chunk_id)} — ${esc(e.where)}</div>
          <div class="evidence-card__quote">"${esc(e.quote)}"</div>
          <div class="evidence-card__supports">Supports: ${esc(e.supports)}</div>
        </div>
      `).join('');
    } else {
      evidenceBody.innerHTML = '<p class="panel__placeholder">No evidence found</p>';
    }

    // Trace
    const ts = data.trace_summary || {};
    traceBody.innerHTML = `
      <div class="trace-item">
        <span class="trace-item__label">Real Switching</span>
        <span class="trace-item__value">${ts.real_switching ? '✓ Yes' : '✗ No'}</span>
      </div>
      <div class="trace-item">
        <span class="trace-item__label">Chunks Processed</span>
        <span class="trace-item__value">${ts.chunks_processed || 0}</span>
      </div>
      <div class="trace-item">
        <span class="trace-item__label">Models Used</span>
        <span class="trace-item__value">${(ts.models_used || []).join(', ') || '—'}</span>
      </div>
      ${renderModeCounts(ts.modes_used_counts || {})}
      <div class="trace-item">
        <span class="trace-item__label">Docs Opened</span>
        <span class="trace-item__value">${(ts.docs_opened || []).join(', ') || '—'}</span>
      </div>
      <div class="trace-item">
        <span class="trace-item__label">Budget</span>
        <span class="trace-item__value">${esc(ts.budget_notes || '—')}</span>
      </div>
      <div class="trace-item">
        <span class="trace-item__label">⚡ Cache Hits</span>
        <span class="trace-item__value" style="color:var(--accent-green)">${data.cache_hits || 0}</span>
      </div>
      <div class="trace-item">
        <span class="trace-item__label">Cache Misses</span>
        <span class="trace-item__value">${data.cache_misses || 0}</span>
      </div>
    `;

    // Confidence
    const conf = data.confidence || 0;
    setConfidence(conf);
  }

  function renderModeCounts(counts) {
    return Object.entries(counts).map(([mode, count]) => {
      const cls = mode.includes('QUICK') ? 'quick' : mode.includes('VISION') ? 'vision' : 'reasoning';
      return `
        <div class="trace-item">
          <span class="trace-item__label">
            <span class="mode-badge mode-badge--${cls}">${mode}</span>
          </span>
          <span class="trace-item__value">${count} calls</span>
        </div>`;
    }).join('');
  }

  // ── Confidence ring ─────────────────────────────────────────
  function setConfidence(value) {
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (value * circumference);
    confRing.style.strokeDashoffset = offset;
    confValue.textContent = Math.round(value * 100) + '%';
  }

  // ── Reset UI ────────────────────────────────────────────────
  function resetUI() {
    stages.forEach(s => {
      stageEls[s].classList.remove('active', 'complete');
      statusEls[s].textContent = 'idle';
    });
    connectors.forEach(c => c.classList.remove('active'));
    answerBody.innerHTML = '<p class="panel__placeholder"><span class="spinner"></span> Processing…</p>';
    evidenceBody.innerHTML = '<p class="panel__placeholder">Waiting…</p>';
    traceBody.innerHTML = '<p class="panel__placeholder">Waiting…</p>';
    confRing.style.strokeDashoffset = 327;
    confValue.textContent = '—';
  }

  // ── Escape HTML ─────────────────────────────────────────────
  function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = String(str);
    return d.innerHTML;
  }

  // ── File Upload ──────────────────────────────────────────────
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');
  const uploadStatus = document.getElementById('uploadStatus');
  const corpusDocs = document.getElementById('corpusDocs');
  const refreshCorpus = document.getElementById('refreshCorpus');

  // Drag-drop events
  ['dragenter', 'dragover'].forEach(ev => {
    dropArea.addEventListener(ev, (e) => {
      e.preventDefault();
      dropArea.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(ev => {
    dropArea.addEventListener(ev, (e) => {
      e.preventDefault();
      dropArea.classList.remove('dragover');
    });
  });

  dropArea.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) uploadFiles(files);
  });

  dropArea.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) uploadFiles(fileInput.files);
    fileInput.value = '';
  });

  refreshCorpus.addEventListener('click', loadCorpus);

  async function uploadFiles(fileList) {
    uploadStatus.innerHTML = '';

    // Disable pipeline while uploading
    runBtn.disabled = true;
    runBtn.style.opacity = '0.5';
    runBtn.style.cursor = 'not-allowed';
    dropArea.style.pointerEvents = 'none';

    // Build step-by-step progress UI
    const steps = [
      { id: 'upload', label: 'Uploading file to server' },
      { id: 'convert', label: 'Converting to text' },
      { id: 'chunk', label: 'Splitting into chunks' },
      { id: 'understand', label: 'AI reading & understanding document' },
      { id: 'ready', label: 'Ready for questions!' },
    ];

    const names = [...fileList].map(f => f.name).join(', ');
    uploadStatus.innerHTML = `
      <div class="upload-steps" style="padding:12px;background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);border-radius:10px;margin-top:8px;">
        <div style="font-weight:600;margin-bottom:10px;color:var(--accent-purple,#a855f7);">
          📄 Processing: ${esc(names)}
        </div>
        ${steps.map((s, i) => `
          <div id="upload-step-${s.id}" class="upload-step" style="
            display:flex;align-items:center;gap:8px;padding:5px 0;
            color:${i === 0 ? 'var(--text-primary,#fff)' : 'var(--text-muted,#888)'};
            transition:color 0.3s, opacity 0.3s;
          ">
            <span id="upload-icon-${s.id}" style="width:20px;text-align:center;">
              ${i === 0 ? '<span class="spinner" style="width:14px;height:14px;"></span>' : '○'}
            </span>
            <span>${s.label}</span>
          </div>
        `).join('')}
      </div>
    `;

    // Animate through steps with estimated timings
    const stepTimers = [];
    const advanceStep = (stepIdx) => {
      if (stepIdx >= steps.length) return;
      // Mark previous step as done
      if (stepIdx > 0) {
        const prev = steps[stepIdx - 1];
        const prevIcon = document.getElementById(`upload-icon-${prev.id}`);
        const prevEl = document.getElementById(`upload-step-${prev.id}`);
        if (prevIcon) prevIcon.innerHTML = '✓';
        if (prevEl) prevEl.style.color = 'var(--accent-green, #4ade80)';
      }
      // Activate current step
      const curr = steps[stepIdx];
      const currIcon = document.getElementById(`upload-icon-${curr.id}`);
      const currEl = document.getElementById(`upload-step-${curr.id}`);
      if (currIcon) currIcon.innerHTML = '<span class="spinner" style="width:14px;height:14px;"></span>';
      if (currEl) currEl.style.color = 'var(--text-primary, #fff)';
    };

    // Estimated step timings (ms) — these animate visually while server processes
    stepTimers.push(setTimeout(() => advanceStep(1), 1500));   // convert
    stepTimers.push(setTimeout(() => advanceStep(2), 3000));   // chunk
    stepTimers.push(setTimeout(() => advanceStep(3), 5000));   // understand (longest)

    const formData = new FormData();
    for (const f of fileList) {
      formData.append('files', f);
    }

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();

      // Clear timers — server responded, upload+chunk+index done
      stepTimers.forEach(t => clearTimeout(t));

      // Mark first 3 steps as complete (upload, convert, chunk — done by server)
      for (let i = 0; i < 3; i++) {
        const icon = document.getElementById(`upload-icon-${steps[i].id}`);
        const el = document.getElementById(`upload-step-${steps[i].id}`);
        if (icon) icon.innerHTML = '✓';
        if (el) el.style.color = 'var(--accent-green, #4ade80)';
      }

      // Show results
      if (data.uploaded && data.uploaded.length) {
        data.uploaded.forEach(u => {
          addStatusItem(u.filename, `Indexed as "${u.doc_id}" (${u.chunks} chunks)`, 'success');
        });
      }
      if (data.errors && data.errors.length) {
        data.errors.forEach(e => {
          addStatusItem(e.filename, e.error, 'error');
        });
      }

      loadCorpus();

      // ── Poll Phase A status for each uploaded doc ──
      const docsToWatch = (data.uploaded || []).filter(u => u.understanding === 'in_progress');
      if (docsToWatch.length > 0) {
        // Activate "understanding" step
        advanceStep(3);

        // Poll until all docs are understood
        const pollInterval = setInterval(async () => {
          let allDone = true;
          for (const doc of docsToWatch) {
            try {
              const statusRes = await fetch(`/api/doc-status/${doc.doc_id}`);
              const statusData = await statusRes.json();
              if (statusData.status !== 'ready') {
                allDone = false;
              }
            } catch (_) {
              allDone = false;
            }
          }

          if (allDone) {
            clearInterval(pollInterval);
            // Mark understand + ready as complete
            const understandIcon = document.getElementById('upload-icon-understand');
            const understandEl = document.getElementById('upload-step-understand');
            if (understandIcon) understandIcon.innerHTML = '✓';
            if (understandEl) understandEl.style.color = 'var(--accent-green, #4ade80)';

            const readyIcon = document.getElementById('upload-icon-ready');
            const readyEl = document.getElementById('upload-step-ready');
            if (readyIcon) readyIcon.innerHTML = '✓';
            if (readyEl) readyEl.style.color = 'var(--accent-green, #4ade80)';

            // Re-enable pipeline
            runBtn.disabled = !queryInput.value.trim();
            runBtn.style.opacity = queryInput.value.trim() ? '1' : '0.5';
            runBtn.style.cursor = queryInput.value.trim() ? '' : 'not-allowed';
            dropArea.style.pointerEvents = '';
          }
        }, 3000); // Poll every 3 seconds
      } else {
        // Already understood — mark all done
        const readyIcon = document.getElementById('upload-icon-ready');
        const readyEl = document.getElementById('upload-step-ready');
        const understandIcon = document.getElementById('upload-icon-understand');
        const understandEl = document.getElementById('upload-step-understand');
        if (understandIcon) understandIcon.innerHTML = '✓';
        if (understandEl) understandEl.style.color = 'var(--accent-green, #4ade80)';
        if (readyIcon) readyIcon.innerHTML = '✓';
        if (readyEl) readyEl.style.color = 'var(--accent-green, #4ade80)';
        // Re-enable
        runBtn.disabled = !queryInput.value.trim();
        runBtn.style.opacity = queryInput.value.trim() ? '1' : '0.5';
        runBtn.style.cursor = queryInput.value.trim() ? '' : 'not-allowed';
        dropArea.style.pointerEvents = '';
      }
      return; // Skip the finally block re-enable — polling handles it

    } catch (err) {
      stepTimers.forEach(t => clearTimeout(t));
      uploadStatus.innerHTML = '';
      addStatusItem('Upload', err.message, 'error');
    } finally {
      // Re-enable pipeline (only if not polling — polling does its own re-enable)
      if (!document.getElementById('upload-icon-understand')?.innerHTML.includes('spinner')) {
        runBtn.disabled = !queryInput.value.trim();
        runBtn.style.opacity = queryInput.value.trim() ? '1' : '0.5';
        runBtn.style.cursor = queryInput.value.trim() ? '' : 'not-allowed';
        dropArea.style.pointerEvents = '';
      }
    }
  }

  function addStatusItem(name, msg, type) {
    const el = document.createElement('div');
    el.className = `upload-status-item upload-status-item--${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : '⟳';
    el.innerHTML = `<strong>${icon} ${esc(name)}</strong> — ${esc(msg)}`;
    uploadStatus.appendChild(el);
  }

  // ── Corpus list ──────────────────────────────────────────────
  async function loadCorpus() {
    try {
      const res = await fetch('/api/corpus');
      const data = await res.json();
      const docs = data.documents || [];

      if (!docs.length) {
        corpusDocs.innerHTML = '<span style="color:var(--text-muted);">No documents in corpus</span>';
        return;
      }

      corpusDocs.innerHTML = docs.map(d => `
        <div class="corpus-doc-chip" data-doc-id="${esc(d.doc_id)}">
          <span class="corpus-doc-chip__name">${esc(d.filename)}</span>
          <span class="corpus-doc-chip__size">${formatSize(d.size)}</span>
          <button class="corpus-doc-chip__delete" title="Remove" onclick="window._deleteDoc('${esc(d.doc_id)}')">×</button>
        </div>
      `).join('');
    } catch (err) {
      corpusDocs.innerHTML = '<span style="color:var(--accent-red);">Failed to load</span>';
    }
  }

  window._deleteDoc = async function (docId) {
    if (!confirm(`Remove "${docId}" from corpus?`)) return;
    try {
      await fetch(`/api/corpus/${docId}`, { method: 'DELETE' });
      loadCorpus();
    } catch (_) { }
  };

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // Load corpus on init
  loadCorpus();

})();
