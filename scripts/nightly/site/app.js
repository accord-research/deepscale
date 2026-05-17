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

// Rows written before the `kind` field existed default to "operational".
function rowKind(r) { return r.kind || 'operational'; }

function renderTrends(metrics) {
  const country = document.getElementById('trend-country').value;
  const season = document.getElementById('trend-season').value;
  const metric = document.getElementById('trend-metric').value;

  const rows = metrics.filter(r =>
    rowKind(r) === 'operational' &&
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
    backgroundColor: colorFor(key),
    spanGaps: false,
    tension: 0.2,
    pointRadius: 4,
    pointHoverRadius: 6,
  }));

  if (chartInstance) chartInstance.destroy();
  const ctx = document.getElementById('trend-chart').getContext('2d');
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {datasets},
    options: {
      responsive: true,
      maintainAspectRatio: false,
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
    if (rowKind(r) !== 'operational') continue;
    if (r.status !== 'ok' || !r.metrics) continue;
    const key = `${r.country}__${r.season}`;
    if (!latestByPair[key] || latestByPair[key].date < r.date) latestByPair[key] = r;
  }
  Object.values(latestByPair).forEach(r => {
    const tr = document.createElement('tr');
    const fmt = (v) => (typeof v === 'number') ? v.toFixed(2) : '–';
    tr.innerHTML = `<td>${esc(r.country)}</td>` +
      `<td>${esc(r.season)}</td>` +
      `<td>${esc(r.init)}</td>` +
      `<td>${fmt(r.metrics.rpss)}</td>` +
      `<td>${fmt(r.metrics.generalized_roc)}</td>` +
      `<td>${fmt(r.metrics['2afc'])}</td>` +
      `<td>${fmt(r.metrics.pearson_r)}</td>` +
      `<td>${fmt(r.metrics.heidke_skill_score)}</td>`;
    tbody.appendChild(tr);
  });
}

function renderForecasts(index) {
  const fcCountry = document.getElementById('fc-country');
  const fcSeason = document.getElementById('fc-season');
  // Default to a country that actually has forecasts in the index. Falls back
  // to the full COUNTRIES list when the index is empty (first-run state).
  const observedCountries = [...new Set(index.map(e => e.country))].sort();
  const countriesToList = observedCountries.length ? observedCountries : COUNTRIES;
  fcCountry.innerHTML = countriesToList.map(c => `<option>${esc(c)}</option>`).join('');

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

let rebenchChartInstance = null;

function renderRebench(metrics) {
  const tupleSel = document.getElementById('rb-tuple');
  const metricSel = document.getElementById('rb-metric');

  const rebenchRows = metrics.filter(r =>
    rowKind(r) === 'rebench' && r.status === 'ok' && r.metrics
  );

  const tuples = [...new Set(rebenchRows.map(
    r => `${r.country}__${r.season}__${r.init}`
  ))].sort();

  // Preserve any prior selection on re-render.
  const prior = tupleSel.value;
  tupleSel.innerHTML = tuples.map(t => {
    const [c, s, i] = t.split('__');
    return `<option value="${esc(t)}">${esc(c)} / ${esc(s)} / init ${esc(i)}</option>`;
  }).join('');
  if (prior && tuples.includes(prior)) tupleSel.value = prior;

  function refresh() {
    const sel = tupleSel.value;
    const metric = metricSel.value;
    if (!sel) {
      if (rebenchChartInstance) { rebenchChartInstance.destroy(); rebenchChartInstance = null; }
      return;
    }
    const [c, s, i] = sel.split('__');
    const series = rebenchRows
      .filter(r => r.country === c && r.season === s && r.init === i)
      .sort((a, b) => a.date.localeCompare(b.date))
      .map(r => ({x: r.date, y: r.metrics[metric] ?? null, commit: r.commit}));

    if (rebenchChartInstance) rebenchChartInstance.destroy();
    const ctx = document.getElementById('rebench-chart').getContext('2d');
    rebenchChartInstance = new Chart(ctx, {
      type: 'line',
      data: {datasets: [{
        label: `${c} ${s} init ${i}`,
        data: series,
        borderColor: colorFor(`${c} ${s} ${i}`),
        backgroundColor: colorFor(`${c} ${s} ${i}`),
        spanGaps: false,
        tension: 0.2,
        pointRadius: 4,
        pointHoverRadius: 6,
      }]},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        scales: {x: {type: 'category'}, y: {title: {display: true, text: metric}}},
        plugins: {tooltip: {callbacks: {
          afterLabel: (ctx) => {
            const p = series[ctx.dataIndex];
            return p ? `commit ${shortSha(p.commit)}` : '';
          }
        }}}
      },
    });
  }
  tupleSel.removeEventListener('change', tupleSel.__rbRefresh || (() => {}));
  metricSel.removeEventListener('change', metricSel.__rbRefresh || (() => {}));
  tupleSel.__rbRefresh = refresh;
  metricSel.__rbRefresh = refresh;
  tupleSel.addEventListener('change', refresh);
  metricSel.addEventListener('change', refresh);
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
  // Derive the season options from the loaded data; fall back to the full
  // 12 standard 3-month windows if metrics.json is empty (first-run state).
  const STANDARD_SEASONS = [
    'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
    'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ',
  ];
  const observedSeasons = [...new Set(metrics.map(r => r.season))].sort();
  const seasonOptions = observedSeasons.length ? observedSeasons : STANDARD_SEASONS;
  trendSeason.innerHTML += seasonOptions.map(s => `<option>${esc(s)}</option>`).join('');

  trendCountry.addEventListener('change', () => renderTrends(metrics));
  trendSeason.addEventListener('change', () => renderTrends(metrics));
  document.getElementById('trend-metric').addEventListener('change', () => renderTrends(metrics));

  renderTrends(metrics);
  renderLatestTable(metrics);
  renderForecasts(index);
  renderRebench(metrics);
  renderRuns(metrics);
}

window.addEventListener('DOMContentLoaded', init);
