// ── State ──
let allClasses = {};
let categories = {};
let activeClasses = new Set();
let availableSources = [];
let currentSource = "webcam";

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
  await loadSources();
  await loadClasses();
  startStatusPolling();
});

// ── Camera Sources ──
async function loadSources() {
  try {
    const res = await fetch('/api/sources');
    const data = await res.json();
    availableSources = data.sources;
    currentSource = data.active;
    renderSources();
  } catch (e) {
    console.error('Failed to load sources', e);
  }
}

function renderSources() {
  const container = document.getElementById('sourceBtns');
  container.innerHTML = '';

  const labels = { webcam: '📷 Webcam', rtsp: '📡 RTSP' };
  const allOptions = ['webcam', 'rtsp'];

  allOptions.forEach(src => {
    const btn = document.createElement('button');
    btn.className = 'source-btn';
    btn.textContent = labels[src] || src;

    if (src === currentSource) btn.classList.add('active');
    if (!availableSources.includes(src)) {
      btn.classList.add('disabled');
      btn.title = 'Not configured';
    } else {
      btn.addEventListener('click', () => switchSource(src));
    }

    container.appendChild(btn);
  });
}

async function switchSource(src) {
  if (src === currentSource) return;
  try {
    const res = await fetch('/api/source', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: src })
    });
    if (res.ok) {
      currentSource = src;
      renderSources();
      // Force-reload the MJPEG stream
      const feed = document.getElementById('videoFeed');
      feed.src = '/video_feed?' + Date.now();
    }
  } catch (e) {
    console.error('Source switch failed', e);
  }
}

// ── Class Filter ──
async function loadClasses() {
  try {
    const res = await fetch('/api/classes');
    const data = await res.json();
    allClasses = data.all_classes;
    categories = data.categories;
    activeClasses = new Set(data.active);
    renderClassFilter();
  } catch (e) {
    console.error('Failed to load classes', e);
  }
}

function renderClassFilter() {
  const container = document.getElementById('classFilter');
  container.innerHTML = '';

  for (const [cat, ids] of Object.entries(categories)) {
    const group = document.createElement('div');
    group.className = 'category-group';

    const label = document.createElement('div');
    label.className = 'category-label';
    label.textContent = cat;
    group.appendChild(label);

    const chips = document.createElement('div');
    chips.className = 'class-chips';

    ids.forEach(id => {
      const chip = document.createElement('label');
      chip.className = 'class-chip' + (activeClasses.has(id) ? ' active' : '');

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = activeClasses.has(id);
      cb.addEventListener('change', () => {
        if (cb.checked) activeClasses.add(id);
        else activeClasses.delete(id);
        chip.classList.toggle('active', cb.checked);
        markDirty();
      });

      chip.appendChild(cb);
      chip.append(allClasses[id]);
      chips.appendChild(chip);
    });

    group.appendChild(chips);
    container.appendChild(group);
  }
}

function selectAll() {
  activeClasses = new Set(Object.keys(allClasses).map(Number));
  renderClassFilter();
  markDirty();
}

function selectNone() {
  activeClasses.clear();
  renderClassFilter();
  markDirty();
}

function markDirty() {
  const btn = document.getElementById('applyBtn');
  btn.textContent = 'Apply';
  btn.classList.remove('saved');
}

async function applyClasses() {
  const btn = document.getElementById('applyBtn');
  try {
    const res = await fetch('/api/classes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ classes: [...activeClasses] })
    });
    if (res.ok) {
      btn.textContent = '✓ Saved';
      btn.classList.add('saved');
      setTimeout(() => {
        btn.textContent = 'Apply';
        btn.classList.remove('saved');
      }, 2000);
    }
  } catch (e) {
    console.error('Failed to apply classes', e);
  }
}

// ── Status Polling ──
function startStatusPolling() {
  pollStatus();
  setInterval(pollStatus, 2000);
}

async function pollStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    document.getElementById('fpsBadge').textContent = `FPS: ${data.fps}`;
    document.getElementById('latencyBadge').textContent = `Latency: ${data.latency}ms`;
    document.getElementById('statusText').textContent =
      `${data.source.toUpperCase()} · ${data.active_classes.length} classes`;
    document.getElementById('statusDot').style.background =
      data.fps > 0 ? 'var(--green)' : 'var(--red)';
  } catch (e) {
    document.getElementById('statusText').textContent = 'Offline';
    document.getElementById('statusDot').style.background = 'var(--red)';
  }
}
