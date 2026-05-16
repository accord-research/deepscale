// scripts/nightly/site/app.js
const COUNTRIES = ['kenya', 'ethiopia', 'nigeria'];

// Escape anything we render via innerHTML so future free-text fields in
// metrics.json (a `reason`, a label) can never execute. publish.py controls
// all current inputs, but this keeps the trust boundary explicit.
function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function shortSha(sha) {
  return String(sha).slice(0, 7);
}

async function loadJSON(path, fallback) {
  try {
    const r = await fetch(path, {cache: 'no-store'});
    if (!r.ok) return fallback;
    return await r.json();
  } catch (e) { return fallback; }
}

function byTab(name) {
  document.querySelectorAll('nav button').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.toggle('active', t.id === 'tab-' + name);
  });
}

document.querySelectorAll('nav button').forEach(b => {
  b.addEventListener('click', () => byTab(b.dataset.tab));
});

let chartInstance = null;

function colorFor(key) {
  // Stable HSL hash per (country, season) key.
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return `hsl(${Math.abs(h) % 360}, 65%, 45%)`;
}

function renderTrends(metrics) {
  const country = document.getElementById('trend-country').value;
  const season = document.getElementById('trend-season').value;
  const metric = document.getElementById('trend-metric').value;

  const rows = metrics.filter(r =>
    (!country || r.country === country) &&
    (!season || r.season === season)
  );

  const groups = {};
  for (const r of rows) {
    const key = `${r.country} ${r.season}`;
    (groups[key] ||= []).push(r);
  }

  const datasets = Object.entries(groups).map(([key, rs]) => ({
    label: key,
    data: rs.map(r => ({
      x: r.date,
      y: r.status === 'ok' && r.metrics ? r.metrics[metric] : null,
    })),
    borderColor: colorFor(key),
    spanGaps: false,
    tension: 0.2,
  }));

  if (chartInstance) chartInstance.destroy();
  const ctx = document.getElementById('trend-chart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {datasets},
    options: {
      parsing: false,
      scales: {x: {type: 'category'}, y: {title: {display: true, text: metric}}},
      plugins: {tooltip: {callbacks: {
        afterLabel: (ctx) => {
          const r = rows.find(rr => rr.date === ctx.parsed.x
                                && `${rr.country} ${rr.season}` === ctx.dataset.label);
          return r ? `commit ${r.commit}` : '';
        }
      }}}
    },
  });
}

function renderLatestTable(metrics) {
  const tbody = document.querySelector('#latest-table tbody');
  tbody.innerHTML = '';
  const latestByPair = {};
  for (const r of metrics) {
    if (r.status !== 'ok' || !r.metrics) continue;
    const key = `${r.country}__${r.season}`;
    if (!latestByPair[key] || latestByPair[key].date < r.date) latestByPair[key] = r;
  }
  Object.values(latestByPair).forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${esc(r.country)}</td><td>${esc(r.season)}</td>` +
      `<td>${(r.metrics.rpss ?? '').toFixed?.(2) ?? '–'}</td>` +
      `<td>${(r.metrics.roc_area ?? '').toFixed?.(2) ?? '–'}</td>` +
      `<td>${(r.metrics.pearson ?? '').toFixed?.(2) ?? '–'}</td>`;
    tbody.appendChild(tr);
  });
}

function renderForecasts(index) {
  const fcCountry = document.getElementById('fc-country');
  const fcSeason = document.getElementById('fc-season');
  fcCountry.innerHTML = COUNTRIES.map(c => `<option>${esc(c)}</option>`).join('');

  function refresh() {
    const c = fcCountry.value;
    const seasons = [...new Set(index.filter(e => e.country === c).map(e => e.season))];
    fcSeason.innerHTML = seasons.map(s => `<option>${esc(s)}</option>`).join('');
    const inits = index
      .filter(e => e.country === c && e.season === fcSeason.value)
      .sort((a, b) => a.init.localeCompare(b.init));
    const strip = document.getElementById('fc-strip');
    strip.innerHTML = inits.map(e => `
      <figure>
        <img src="forecasts/${encodeURIComponent(e.country)}/${encodeURIComponent(e.season)}/${encodeURIComponent(e.init)}/tercile_map.png"
             alt="${esc(e.country)} ${esc(e.season)} ${esc(e.init)}">
        <figcaption>Init ${esc(e.init)}</figcaption>
      </figure>
    `).join('');
  }
  fcCountry.addEventListener('change', refresh);
  fcSeason.addEventListener('change', refresh);
  refresh();
}

function renderRuns(metrics) {
  const list = document.getElementById('run-list');
  const byDate = {};
  for (const r of metrics) (byDate[r.date] ||= []).push(r);
  const dates = Object.keys(byDate).sort().reverse();
  list.innerHTML = dates.map(d => {
    const rs = byDate[d];
    const commit = rs[0].commit;
    const badges = COUNTRIES.map(c => {
      const country_rows = rs.filter(r => r.country === c);
      const ok = country_rows.length > 0 && country_rows.every(r => r.status === 'ok');
      return `<span class="badge ${ok ? 'ok' : 'fail'}">${ok ? '✓' : '✗'} ${esc(c.slice(0,2).toUpperCase())}</span>`;
    }).join(' ');
    return `<li><time>${esc(d)}</time> commit <code>${esc(shortSha(commit))}</code> ${badges}</li>`;
  }).join('');
}

async function init() {
  const metrics = await loadJSON('metrics.json', []);
  const index = await loadJSON('forecasts/index.json', []);

  const latest = metrics.length ? metrics[metrics.length - 1] : null;
  if (latest) {
    document.getElementById('last-run').textContent =
      `last run: ${latest.date}   commit: ${shortSha(latest.commit)}`;
  }

  const trendCountry = document.getElementById('trend-country');
  trendCountry.innerHTML += COUNTRIES.map(c => `<option>${c}</option>`).join('');
  const trendSeason = document.getElementById('trend-season');
  trendSeason.innerHTML += ['MAM', 'JJAS', 'OND'].map(s => `<option>${s}</option>`).join('');

  trendCountry.addEventListener('change', () => renderTrends(metrics));
  trendSeason.addEventListener('change', () => renderTrends(metrics));
  document.getElementById('trend-metric').addEventListener('change', () => renderTrends(metrics));

  renderTrends(metrics);
  renderLatestTable(metrics);
  renderForecasts(index);
  renderRuns(metrics);
}

window.addEventListener('DOMContentLoaded', init);
