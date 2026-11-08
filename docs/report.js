'use strict';
(() => {
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const byGene = new Map(data.rows.map(row => [row.gene, row]));
  const $ = id => document.getElementById(id);
  const themeToggle = $('theme-toggle');
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeToggle.textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
    themeToggle.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
    try { localStorage.setItem('alz-report-theme', theme); } catch {}
  }
  applyTheme(document.documentElement.dataset.theme || 'dark');
  themeToggle.addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  function openDisclosure(id) { const target = document.getElementById(id); if (target?.tagName === 'DETAILS') target.open = true; }
  document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', () => openDisclosure(link.hash.slice(1))));
  openDisclosure(location.hash.slice(1));
  const pageSize = 12;
  let page = 0;
  let selected = 'CLEC5A';
  const signed = value => value == null ? '—' : `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(3)}`;
  const fdr = value => value == null ? '—' : value < .0001 ? value.toExponential(2) : value.toFixed(4);
  const percentage = value => `${(value * 100).toFixed(0)}%`;
  const make = (tag, text, attrs = {}) => {
    const el = document.createElement(tag);
    if (text != null) el.textContent = text;
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    return el;
  };
  function filteredRows() {
    const query = $('gene-search').value.trim().toUpperCase();
    const filter = $('gene-filter').value;
    const rows = data.rows.filter(row => row.gene.toUpperCase().includes(query) && (filter === 'all' || (filter === 'robust' && row.robust_sensitivity) || (filter === 'cohort' && row.rosmap_support) || (filter === 'stable' && row.stable)));
    const sort = $('gene-sort').value;
    rows.sort((a, b) => sort === 'gene' ? a.gene.localeCompare(b.gene) : sort === 'effect' ? Math.abs(b.model_effect) - Math.abs(a.model_effect) : sort === 'fdr' ? (a.rosmap_fdr ?? Infinity) - (b.rosmap_fdr ?? Infinity) || a.rank - b.rank : a.rank - b.rank);
    return rows;
  }
  function drawTable() {
    const rows = filteredRows();
    const pages = Math.max(1, Math.ceil(rows.length / pageSize));
    page = Math.min(page, pages - 1);
    const shown = rows.slice(page * pageSize, (page + 1) * pageSize);
    const body = $('gene-rows');
    body.replaceChildren();
    for (const row of shown) {
      const tr = make('tr');
      if (row.gene === selected) tr.classList.add('selected');
      const th = make('th', null, {scope: 'row'});
      const button = make('button', row.gene, {type: 'button', 'aria-label': `Inspect ${row.gene}`, 'aria-pressed': String(row.gene === selected)});
      button.addEventListener('click', () => { selected = row.gene; drawDetail(); drawTable(); });
      th.append(button); tr.append(th);
      for (const value of [signed(row.model_effect), signed(row.real_effect), signed(row.adjusted_effect), fdr(row.rosmap_fdr)]) tr.append(make('td', value));
      body.append(tr);
    }
    if (!rows.length) { const tr = make('tr'); tr.append(make('td', 'No genes match this search and filter.', {colspan: '5'})); body.append(tr); }
    $('gene-count').textContent = rows.length ? `${rows.length.toLocaleString()} genes · showing ${page * pageSize + 1}–${Math.min((page + 1) * pageSize, rows.length)} · Δ = High minus Not AD` : '0 genes · Try “All measured genes” to broaden the filter.';
    $('page-state').textContent = `${page + 1} / ${pages}`;
    $('previous-page').disabled = page === 0;
    $('next-page').disabled = page >= pages - 1;
  }
  function drawProfile(gene) {
    const profile = data.stage_profiles[gene];
    const means = [0, 1, 2, 3].map(i => profile.model_by_seed.reduce((sum, row) => sum + row[i], 0) / 5);
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 460 275'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${gene} mean expression across Not AD, Low, Intermediate and High pathology. Observed: ${profile.real.map(v => v.toFixed(3)).join(', ')}. Model: ${means.map(v => v.toFixed(3)).join(', ')}. Units: mean log1p panel CP10k.`);
    const shape = (tag, attrs, text) => { const el = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); if (text != null) el.textContent = text; svg.append(el); return el; };
    const top = Math.max(...profile.real, ...profile.model_by_seed.flat(), .01) * 1.14;
    const x = i => 55 + i * 120;
    const y = value => 224 - value / top * 180;
    shape('text', {x: 55, y: 17, fill: 'var(--dl-fg-muted)', 'font-size': 11}, 'Mean log1p(panel CP10k)');
    for (let i = 0; i <= 4; i++) { const value = top * i / 4; shape('line', {x1: 55, x2: 415, y1: y(value), y2: y(value), stroke: 'var(--dl-border)', 'stroke-width': .7}); shape('text', {x: 45, y: y(value) + 4, fill: 'var(--dl-fg-muted)', 'font-size': 10, 'text-anchor': 'end'}, value.toFixed(top < .1 ? 3 : 2)); }
    ['Not AD', 'Low', 'Interm.', 'High'].forEach((label, i) => shape('text', {x: x(i), y: 247, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--dl-fg-muted)'}, label));
    const line = (values, color, width, opacity) => shape('polyline', {points: values.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: color, 'stroke-width': width, opacity});
    for (const seed of profile.model_by_seed) line(seed, 'var(--dl-accent)', 1.2, .23);
    line(means, 'var(--dl-accent)', 2.6, 1); line(profile.real, 'var(--dl-fg)', 2, 1);
    means.forEach((v, i) => shape('circle', {cx: x(i), cy: y(v), r: 3.5, fill: 'var(--dl-accent)'}));
    profile.real.forEach((v, i) => shape('rect', {x: x(i) - 3, y: y(v) - 3, width: 6, height: 6, fill: 'var(--dl-fg)'}));
    $('detail-chart').replaceChildren(svg);
  }
  function drawDetail() {
    const row = byGene.get(selected);
    $('detail-gene').textContent = selected;
    $('detail-status').textContent = row.robust_sensitivity ? 'Passes sensitivity filter' : row.rosmap_support ? 'Model + cohort support' : row.stable ? 'Stable model signal' : 'Outside stability filter';
    $('detail-summary').textContent = `Original within-subtype heuristic rank ${row.rank.toLocaleString()} of 2,000. Model Δ ${signed(row.model_effect)}; observed Δ ${signed(row.real_effect)}; adjusted observed Δ ${signed(row.adjusted_effect)}.`;
    drawProfile(selected);
    const metrics = [ ['Seeds agreeing with observed direction', `${row.seeds_agree} / 5`], ['Monte Carlo sign agreement', percentage(row.mc_agreement)], ['ROSMAP pathology coefficient', signed(row.rosmap_logFC)], ['ROSMAP published global FDR', fdr(row.rosmap_fdr)], ['Minimum adjusted control-deletion agreement', percentage(row.minimum_control_deletion_agreement)], ['Model seed-effect range', `${signed(Math.min(...row.seed_effects))} to ${signed(Math.max(...row.seed_effects))}`] ];
    const dl = $('detail-metrics'); dl.replaceChildren();
    for (const [label, value] of metrics) { const div = make('div'); div.append(make('dt', label), make('dd', value)); dl.append(div); }
    const caveats = {
      CLEC5A: 'All five endpoint effects are positive, but the observed Low-stage peak is missed by the mean model. Mouse knockout results motivate reduction; the model does not predict that intervention’s benefit.',
      CD163: 'The local association reverses after covariate adjustment. The generator also substantially underestimates absolute expression. This is not a robust nominated target.',
      USP9Y: 'This Y-linked gene is sensitive to sex imbalance. The adjusted association reverses. A reproducible model signal can reproduce cohort confounding.',
      UTY: 'This Y-linked association reverses after donor-covariate adjustment. Sex imbalance is a major interpretive concern.',
      CH25H: 'One control-donor deletion reverses the association in one eligible subtype. It fails the additional stage-specific donor robustness filter.',
      GPNMB: 'The model strongly attenuates the endpoint effect. Higher disease-associated expression does not imply that reducing this potentially adaptive state is beneficial.',
      PPARG: 'The model underestimates absolute expression. The direction of a useful intervention is unresolved; the low-dose pioglitazone prevention trial had a negative primary result.',
      LPL: 'The small positive endpoint association is consistent, but the stage shape is imperfect. Association does not specify whether activity should be increased or reduced.',
      PLIN2: 'A candidate lipid-droplet-associated readout, with attenuated model effect. A marker change alone is not evidence of improved microglial function.'
    };
    $('detail-caveat').textContent = caveats[selected] || 'Exploratory observational evidence only. Model-seed agreement measures simulation consistency, not biological replication. The external coefficient has a different scale; neither comparison predicts an intervention effect.';
  }
  for (const id of ['gene-search', 'gene-filter', 'gene-sort']) $(id).addEventListener(id === 'gene-search' ? 'input' : 'change', () => { page = 0; drawTable(); });
  $('previous-page').addEventListener('click', () => { page--; drawTable(); });
  $('next-page').addEventListener('click', () => { page++; drawTable(); });
  drawTable(); drawDetail();
})();
